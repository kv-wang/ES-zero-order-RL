#!/usr/bin/env python
"""Axis-probe trainer -- three parallel, manually selected variants.

  --variant momentum : 4 probe members (2 antithetic pairs along the EMA momentum m_t)
                       + (N-4) Gaussian.                                    [A1, new]
  --variant baseaxis : 4 probe members along theta0 - theta_t + (N-4) Gaussian.
                       Byte-equivalent to 2026-07-23 base_axis_probe.       [baseline 1]
  --variant vanilla  : N Gaussian members, nothing else.                    [baseline 0]

Everything outside the axis choice is shared: binary reward, CRN batches, shared z-score pool,
2 antithetic pairs with radii a1,a2 ~ U(0,a_max], mixed central-difference commit.

-------------------------------------------------------------------------------
Probe coefficient -- one derivation covering both axes
-------------------------------------------------------------------------------
A Gaussian member perturbs by p_i = sigma*eps_i and contributes (alpha/N)*z_i*eps_i, i.e.
(alpha/(N*sigma)) * z_i * p_i. Applying the same rule to a probe member whose perturbation is
c*v for a signed scalar c gives, after the antithetic pair cancels the pool mean,

    Delta_probe = (alpha/(N*sigma)) * sum_k a_k * scale * (z_k+ - z_k-) * v
    member coef = sign * a_k * scale

The only difference between the two axes is `scale`:

  baseaxis  scale = 1
            a_k interpolates directly toward theta0; a=1 lands exactly ON theta0, a natural
            landmark. Perturbation norm is a_k*||theta0-theta_t||, which grows with drift.

  momentum  scale = sigma*sqrt(d) / ||m||
            m_t has no landmark at a=1, so the axis is renormalised to the Gaussian
            perturbation norm. a_k is then read as "fraction of one Gaussian member's step",
            and a_max=1.0 probes exactly as far as a typical Gaussian member. Without this
            the 2026-07-23 smoke failure repeats: a probe 27-123x below sigma*sqrt(d) makes
            f+ - f- identically zero.

What momentum-vs-baseaxis does and does NOT isolate
---------------------------------------------------
Held identical: training batches (drawn from C.DATA_SEED, independent of pop_seed), the reward,
the eval battery, the population size, and -- because the two gates are aligned (see
--mom_warmup) -- the prng stream, so both arms draw the same radii a1,a2 and the same 12
Gaussian seeds at every step. The comparison is therefore close to paired.

NOT isolated: the radius policy differs by necessity (`scale` above). Measured on 07-23 seed 0,
the baseaxis probe norm grows 2.6 -> 30.1 over 200 steps while momentum is pinned near
a*sigma*sqrt(d) ~ 19.6 on average. Early in training momentum probes considerably further than
baseaxis; late they are comparable. Two things differ between the arms, the axis and its scale,
and no run can separate them. Report it as such.

Note vanilla is NOT prng-matched to either probe arm: it draws 16 seeds per step and no radii,
so its Gaussian stream diverges from step 1. That was already true of the 07-23 vanilla control.

Deliberately NOT changed from the 2026-07-23 implementation:
  - a1, a2 drawn i.i.d. (they can collide; both radii are logged so this is measurable)
  - probe members are skipped, not zeroed, below the activation gate (population composition
    changes at the gate; `use_probe` is logged every step)
  - binary reward, no continuous tiebreaker (~67% of pairs tie at B=8)

Added, identically for all three arms, so neither arm gains an advantage:
  - per-member fitness / z / seeds in the JSONL (the 07-23 trainer logged aggregates only,
    which made every post-hoc reanalysis impossible)
  - signed_disp_frac = signed_disp / axis_norm (the raw cum_disp is not comparable across
    steps because ||axis|| grows during training)
  - correct_bits: per-member per-question correctness as a bitmask (added 2026-08-10).
    Without it split-half Spearman cannot be computed, and the 2026-08-09 run therefore
    could not say whether MATH L3-5 carries any extractable member-ranking signal --
    the very question that would explain its null results.

The 2026-08-09 run of this trainer used the pre-fix \\boxed extractor and its results are
archived under axis_probe/archive_pre_extractor_fix_20260809/. See EXTRACTOR_BUG_REPORT.md.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np


def kappa_to_lambda(kappa: float, d: float) -> float:
    """λ = sqrt(κ·d/(1-κ)). Caps κ→1 to avoid overflow."""
    kappa = float(kappa)
    if kappa <= 0.0:
        return 0.0
    if kappa >= 0.9999:
        return 100.0
    return math.sqrt(kappa * d / (1.0 - kappa))


def lambda_to_kappa(lam: float, d: float) -> float:
    """κ = λ²/(d+λ²), inverse of kappa_to_lambda for finite λ."""
    lam = float(lam)
    if lam <= 0.0:
        return 0.0
    lam2 = lam * lam
    return lam2 / (d + lam2)


def tilt_cosine_lambda(step: int, num_steps: int, lam_max: float, lam_min: float) -> float:
    """Cosine decay of λ from lam_max (step 0) to lam_min (final step).

    λ(t) = λ_min + ½(λ_max-λ_min)(1 + cos(π·t/(T-1))), matching common LR cosine schedules.
    """
    if num_steps <= 1:
        return float(lam_min)
    progress = step / (num_steps - 1)  # 0 at first step, 1 at last
    progress = min(1.0, max(0.0, progress))
    return lam_min + 0.5 * (lam_max - lam_min) * (1.0 + math.cos(math.pi * progress))


def tilt_half_vanilla_kappa(step: int, num_steps: int, kappa0: float, switch_frac: float = 0.5) -> float:
    """Piecewise schedule: κ=κ0 for the first switch_frac of steps, then κ=0 (vanilla ES).

    With num_steps=100 and switch_frac=0.5: steps 0..49 use tilt, steps 50..99 use vanilla.
    """
    if num_steps <= 0:
        return 0.0
    switch_frac = min(1.0, max(0.0, float(switch_frac)))
    switch_step = int(num_steps * switch_frac)  # first step that is vanilla
    return float(kappa0) if step < switch_step else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["momentum", "baseaxis", "vanilla", "tilt"], required=True)
    p.add_argument("--population_size", type=int, required=True)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--pop_seed", type=int, default=0)
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU id(s): single '0' or comma-separated '0,1' for tensor parallelism")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="number of GPUs for tensor parallelism (must match --gpu count)")
    p.add_argument("--a_max", type=float, default=1.0)
    p.add_argument("--anchor_threshold", type=float, default=1.0,
                   help="baseaxis only: min ||theta0-theta_t|| to activate probes")
    p.add_argument("--mom_beta", type=float, default=0.9,
                   help="momentum only: EMA decay; effective window ~1/(1-beta)")
    p.add_argument("--mom_warmup", type=int, default=1,
                   help="momentum only: steps of pure Gaussian ES before probes activate. "
                        "Default 1 aligns the momentum gate with the baseaxis gate: measured "
                        "on all five 07-23 runs, ||theta0-theta_t|| crosses anchor_threshold=1.0 "
                        "at step 1 and stays above it (199/200 steps active). Matching the gate "
                        "keeps the two probe arms on an IDENTICAL prng stream -- same radii, "
                        "same Gaussian seeds, every step -- so momentum-vs-baseaxis is paired.")
    # --- tilt variant: rank-one momentum tilt for ALL members ---
    p.add_argument("--tilt_kappa", type=float, default=0.0,
                   help="tilt only: energy fraction in momentum direction. 0=vanilla, 0.2 typical. "
                        "Perturbation covariance C = σ²(I + λ²·m̂·m̂ᵀ) where λ=sqrt(κ·d/(1-κ)). "
                        "κ=0 disables tilt (pure Gaussian ES), κ→1 collapses to 1D. "
                        "With --tilt_lambda_cosine this is the INITIAL κ (maps to λ_max). "
                        "With --tilt_half_vanilla this is κ for the first half of steps.")
    p.add_argument("--tilt_kappa_end", type=float, default=0.0,
                   help="tilt only: final energy fraction when --tilt_lambda_cosine is set "
                        "(maps to λ_min). Default 0 = anneal to vanilla ES.")
    p.add_argument("--tilt_lambda_cosine", action="store_true",
                   help="tilt only: cosine-decay λ from λ(κ=tilt_kappa) to λ(κ=tilt_kappa_end) "
                        "over num_steps, then set κ=λ²/(d+λ²) each step. Prefer this over "
                        "cosine on κ itself (κ↔λ is nonlinear). Mutually exclusive with "
                        "--tilt_half_vanilla.")
    p.add_argument("--tilt_half_vanilla", action="store_true",
                   help="tilt only: use κ=tilt_kappa for the first --tilt_switch_frac of steps, "
                        "then κ=0 (vanilla ES) for the remainder. Mutually exclusive with "
                        "--tilt_lambda_cosine.")
    p.add_argument("--tilt_switch_frac", type=float, default=0.5,
                   help="tilt only: with --tilt_half_vanilla, fraction of steps that use tilt "
                        "before switching to vanilla. Default 0.5 = first half.")
    p.add_argument("--tilt_mom_beta", type=float, default=0.9,
                   help="tilt only: EMA decay for momentum buffer m = β·m + Δ")
    p.add_argument("--tilt_warmup", type=int, default=1,
                   help="tilt only: steps of vanilla ES before tilt activates (m needs history)")
    # --- adaptive sigma (2026-08-10): sigma scales WITH the ES gradient norm ---
    p.add_argument("--sigma_adapt", action="store_true",
                   help="scale sigma with the ES gradient norm. alpha is left untouched.")
    p.add_argument("--sigma_adapt_gain", type=float, default=1.0,
                   help="exponent on the gradient-norm ratio; 1.0 = proportional")
    p.add_argument("--sigma_adapt_clip", type=float, default=2.0,
                   help="sigma is confined to [sigma0/clip, sigma0*clip]")
    p.add_argument("--sigma_adapt_ema", type=float, default=0.9,
                   help="EMA decay on fitness_std; smooths the ~17%% zero-update steps")
    p.add_argument("--sigma_adapt_warmup", type=int, default=10,
                   help="steps at sigma0 used to calibrate the reference gradient norm")
    # --- continuous reward (2026-09-13): format+answer instead of binary correctness ---
    p.add_argument("--continuous_reward", action="store_true",
                   help="countdown only: use 0.1×format + answer ∈ [0,1.1] instead of binary {0,1}. "
                        "Provides finer gradient signal when correctness is sparse.")
    p.add_argument("--kl", action="store_true")
    p.add_argument("--eval_final", action="store_true")
    p.add_argument("--eval_interval", type=int, default=None,
                   help="Evaluate every N steps during training. None = no intermediate eval.")
    p.add_argument("--gpu_mem_util", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization. Peak-memory numbers are NOT comparable "
                        "across different values of this: vLLM sizes the KV cache to fill the "
                        "budget, so the measured peak reflects the setting, not the requirement. "
                        "Set it equal to the GRPO arm's rollout.gpu_memory_utilization (0.5) "
                        "before comparing the two methods.")
    p.add_argument("--eval_cap", type=int, default=300)
    # Countdown needs a different data source, prompt form (raw `context`, no chat template)
    # and reward (countdown_task.answer_reward_function), but the SAME ES loop -- so it is a
    # data/reward switch here, not a second trainer. B/sigma/alpha become overridable because
    # the countdown battery runs at N=30 B=100, not the math defaults N=16 B=8.
    p.add_argument("--dataset", default=None, help="default: config.DATASET; 'countdown' switches task")
    p.add_argument("--batch", type=int, default=None, help="B, prompts per member; default config.BATCH_SIZE")
    p.add_argument("--mini_batch", type=int, default=None,
                   help="vLLM generate chunk size; default config.MINI_BATCH_SIZE")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, HERE)
    import config as C

    N = a.population_size
    num_steps = a.num_steps or C.NUM_STEPS
    probe_variant = a.variant in ("momentum", "baseaxis")
    tilt_variant = a.variant == "tilt"
    if probe_variant and N <= 4:
        raise SystemExit(f"{a.variant} needs N>4 (4 probe members + >=1 Gaussian)")
    if a.tilt_lambda_cosine and a.tilt_half_vanilla:
        raise SystemExit("--tilt_lambda_cosine and --tilt_half_vanilla are mutually exclusive")
    if a.tilt_half_vanilla and not (0.0 < a.tilt_switch_frac <= 1.0):
        raise SystemExit("--tilt_switch_frac must be in (0, 1]")
    which = {"momentum": "mom", "baseaxis": "base", "vanilla": None, "tilt": None}[a.variant]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench import data_math

    dataset = a.dataset or C.DATASET
    B = a.batch or C.BATCH_SIZE
    mini_batch = a.mini_batch or getattr(C, "MINI_BATCH_SIZE", B)
    sigma = a.sigma if a.sigma is not None else C.SIGMA
    alpha = a.alpha if a.alpha is not None else C.ALPHA
    max_tokens = a.max_tokens or C.MAX_TOKENS
    warmup = C.WARMUP
    tok = AutoTokenizer.from_pretrained(C.MODEL)

    # Same switch as 07-24's es_train_fullparam.py, so the two trainers agree on what
    # "countdown" means: rows [300:] of countdown.json (the first 300 are the pinned eval
    # slice, so train and eval are disjoint by construction), the dataset's own raw `context`
    # as the prompt -- NO chat template, matching the original ES setup and the eval path --
    # and the countdown_task reward. By default this is binary answer_reward_function (0/1);
    # --continuous_reward switches to 0.1×format + answer ∈ [0,1.1].
    if dataset == "countdown":
        from es_bench import data_countdown as dc
        train, _ = dc.load_split()
        mk_prompt = lambda r: r["context"]
        if a.continuous_reward:
            row_reward = lambda text, r: dc.reward_continuous(text, r)
        else:
            row_reward = lambda text, r: dc.reward(text, r)
    else:
        levels = None if dataset == "gsm8k" else C.LEVELS
        train, _ = data_math.make_split(dataset, C.TRAIN_SIZE, 200, C.DATA_SEED, levels=levels)
        mk_prompt = lambda r: data_math.build_prompt(tok, r["question"])
        row_reward = lambda text, r: math_reward(text, r["gt"])["reward"]
    drng = random.Random(C.DATA_SEED)
    step_batches = [[drng.randrange(len(train)) for _ in range(B)] for _ in range(num_steps)]

    llm = LLM(model=C.MODEL, dtype=C.DTYPE, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              tensor_parallel_size=a.tensor_parallel_size,
              worker_extension_cls="es_bench.axis_probe.src.axis_worker.AxisWorker")
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42, stop=C.STOP)
    llm.collective_rpc("es_snapshot_base")
    if which == "base":
        llm.collective_rpc("es_snapshot_theta0")
    elif which == "mom":
        llm.collective_rpc("es_mom_init")
    elif tilt_variant:
        llm.collective_rpc("es_mom_init")  # tilt also needs momentum buffer
    d = llm.collective_rpc("es_param_count")[0]
    sigma_scale = sigma * math.sqrt(d)            # Gaussian perturbation norm ~ sigma*sqrt(d)

    kl_rec = None
    if a.kl:
        import kl as klmod
        kl_prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
        kl_rec = klmod.capture_base(llm, kl_prompts, max_tokens=C.MAX_TOKENS)
        print(f"[kl] base-vs-base drift={klmod.drift(llm, tok, kl_rec):.2e}", flush=True)

    # ---- adaptive-sigma controller state ----
    # Signal: fitness_std, which IS the ES gradient norm up to a known constant. With
    # eps_i near-orthogonal in d dimensions,
    #     ||g_hat|| = ||(1/(N*sigma)) sum_i (f_i - f_bar) eps_i|| ~ sqrt(d/N) * std(f) / sigma
    # so std(f) and ||g_hat|| differ only by sqrt(d/N)/sigma. Unlike the probe's
    # sum_k a_k (z+ - z-), std(f) is defined for the vanilla arm too, and unlike ||Delta||
    # (which z-scoring pins to the constant alpha*sqrt(d/N)) it actually varies:
    # measured coefficient of variation 0.59-0.64 across the 2026-08-10 arms.
    #
    # The controller is CAUSAL: sigma for step t is chosen from fitness_std observed at
    # steps < t, because members must be perturbed before their fitness exists.
    fs_ema = None
    fs_ref = None
    fs_warm = []
    sigma0 = sigma

    llm.collective_rpc("es_reset_peak_mem")   # measure the TRAINING loop, not model load
    prng = random.Random(a.pop_seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jf = open(a.out_prefix + ".jsonl", "w")
    cum_disp = 0.0
    cum_disp_frac = 0.0
    t_run0 = time.perf_counter()

    # Intermediate eval tracking. eval_history is also flushed to its own JSONL after every
    # intermediate eval, not just into the summary at the end: a 100-step arm is ~1.4 h and any
    # kill (OOM, preemption, Ctrl-C) would otherwise discard every curve point already paid for.
    eval_history = []  # will hold {step: int, eval_result: dict, eval_seconds: float}
    ef = open(a.out_prefix + "_evalcurve.jsonl", "w") if a.eval_interval else None

    def member_primary(prompts, rows):
        """Returns (mean reward, tokens, per-question correctness as a bitmask).

        The bitmask (bit b = member solved problem b of this step's CRN batch) is what
        makes split-half Spearman computable post hoc: split the B problems into halves,
        score each member on each half, correlate the two rankings. The 2026-08-09 run
        logged only the mean and therefore could not answer whether MATH L3-5 carries any
        extractable member-ranking signal at all -- the question that would explain its
        null results. correct_bits is one Python int per member (fine for B<=1024).
        """
        rewards, toks = [], 0
        chunk = max(1, mini_batch)
        for start in range(0, len(prompts), chunk):
            p_chunk = prompts[start:start + chunk]
            r_chunk = rows[start:start + chunk]
            outs = llm.generate(p_chunk, sp, use_tqdm=False)
            rewards.extend(row_reward(o.outputs[0].text, r) for o, r in zip(outs, r_chunk))
            toks += int(sum(len(o.outputs[0].token_ids) for o in outs))
        bits = 0
        for b, r in enumerate(rewards):
            if r >= 1.0:
                bits |= (1 << b)
        return float(np.mean(rewards)), toks, bits

    for step in range(num_steps):
        # ---- adaptive sigma: bigger ES gradient norm -> bigger perturbation ----
        sigma = sigma0
        sigma_ratio = 1.0
        if a.sigma_adapt and fs_ref is not None and fs_ref > 0 and fs_ema is not None:
            raw = (fs_ema / fs_ref) ** a.sigma_adapt_gain
            lo, hi = 1.0 / a.sigma_adapt_clip, a.sigma_adapt_clip
            sigma_ratio = min(hi, max(lo, raw))
            sigma = sigma0 * sigma_ratio
        sigma_scale = sigma * math.sqrt(d)

        batch = [train[i] for i in step_batches[step]]
        prompts = [mk_prompt(b) for b in batch]

        # ---- tilt variant: kappa -> lambda, gated on ||m|| ----
        # Schedules (optional): cosine decay on λ, or piecewise half-tilt / half-vanilla.
        if tilt_variant:
            if a.tilt_lambda_cosine:
                lam_sched = tilt_cosine_lambda(
                    step, num_steps,
                    kappa_to_lambda(a.tilt_kappa, d),
                    kappa_to_lambda(a.tilt_kappa_end, d),
                )
                kappa = lambda_to_kappa(lam_sched, d)
            elif a.tilt_half_vanilla:
                kappa = tilt_half_vanilla_kappa(
                    step, num_steps, a.tilt_kappa, a.tilt_switch_frac)
                lam_sched = kappa_to_lambda(kappa, d)
            else:
                kappa = a.tilt_kappa
                lam_sched = kappa_to_lambda(kappa, d)
            # The worker folds 1/||m|| into its scalar coefficients, so the gate only needs
            # the norm -- a materialized m_hat would be a third full fp32 copy of the model,
            # which this card has no room for. ||m||=0 (nothing committed yet) disables tilt.
            mom_norm = llm.collective_rpc("es_axis_norm", args=("mom",))[0]
            use_tilt = (step >= a.tilt_warmup) and (kappa > 0.0) and (mom_norm > 0.0)
            lam = lam_sched if use_tilt else 0.0
        else:
            use_tilt = False
            lam = 0.0
            kappa = 0.0
            mom_norm = 0.0

        # ---- axis state and activation gate ----
        axis_norm0 = llm.collective_rpc("es_axis_norm", args=(which,))[0] if which else 0.0
        if which == "base":
            use_probe = axis_norm0 > a.anchor_threshold
            scale = 1.0
        elif which == "mom":
            use_probe = (step >= a.mom_warmup) and (axis_norm0 > 0.0)
            scale = (sigma_scale / axis_norm0) if axis_norm0 > 0.0 else 0.0
        else:
            use_probe, scale = False, 0.0

        # ---- population composition: 2 antithetic pairs, then Gaussians ----
        anchors = []   # (a_k, sign) in fixed order: +a1,-a1,+a2,-a2
        if use_probe:
            a1 = prng.uniform(0, a.a_max); a2 = prng.uniform(0, a.a_max)
            anchors = [(a1, +1), (a1, -1), (a2, +1), (a2, -1)]
        n_gauss = N - len(anchors)
        gauss_seeds = [prng.randrange(2**31 - 1) for _ in range(n_gauss)]

        fitness, member_bits, step_tokens = [], [], 0

        # Tilt variant: all N members are tilted Gaussians, no probe.
        # Reuse the gauss_seeds drawn above (same RNG stream as vanilla when use_probe=False /
        # lam=0) -- do NOT draw a second batch of seeds and discard the first.
        if tilt_variant:
            for seed in gauss_seeds:
                llm.collective_rpc("es_set_tilt_member", args=(seed, sigma, lam, mom_norm))
                f_, tk, bits = member_primary(prompts, batch)
                fitness.append(f_); member_bits.append(bits); step_tokens += tk
        else:
            # Probe variants: anchors first, then Gaussians
            for (av, sgn) in anchors:
                llm.collective_rpc("es_set_axis_member", args=(sgn * av * scale, which))
                f_, tk, bits = member_primary(prompts, batch)
                fitness.append(f_); member_bits.append(bits); step_tokens += tk
            for seed in gauss_seeds:
                llm.collective_rpc("es_set_member", args=(seed, sigma))
                f_, tk, bits = member_primary(prompts, batch)
                fitness.append(f_); member_bits.append(bits); step_tokens += tk

        f = np.array(fitness, dtype=np.float64)
        z = (f - f.mean()) / (f.std() + 1e-8)

        # Tilt variant: all z values are from tilted Gaussians
        if tilt_variant:
            gauss_z = z
            # Same convention as the vanilla path: the worker rebuilds the sigma-FREE direction
            # (xi + lam*zeta*m/||m||), so c carries alpha/N and the estimator's 1/sigma stays
            # implicit. Tilt changes the DIRECTION members are drawn along, never the step scale.
            gauss_coeffs = ((alpha / N) * gauss_z).tolist()
            s_axis = 0.0
            pair_fdiff = []
            n_eff = float((z.sum() ** 2) / (z ** 2).sum()) if (z ** 2).sum() > 0 else 0.0
        else:
            gauss_z = z[len(anchors):]
            gauss_coeffs = ((alpha / N) * gauss_z).tolist()
            s_axis = 0.0; pair_fdiff = []
            if use_probe:
                for (pi, av) in [((0, 1), anchors[0][0]), ((2, 3), anchors[2][0])]:
                    p_plus, p_minus = pi
                    s_axis += av * (z[p_plus] - z[p_minus])
                    pair_fdiff.append(float(f[p_plus] - f[p_minus]))
                s_axis *= (alpha / N) / sigma * scale
            n_eff = 0.0

        # Build tilt_params dict if tilt is active
        tilt_params = None
        if tilt_variant and use_tilt:
            tilt_params = {"lam": lam, "mom_norm": mom_norm}

        # Determine momentum beta: tilt uses tilt_mom_beta, momentum axis uses mom_beta
        mom_beta_arg = None
        if tilt_variant:
            mom_beta_arg = a.tilt_mom_beta
        elif which == "mom":
            mom_beta_arg = a.mom_beta

        res = llm.collective_rpc(
            "es_commit_update_axis",
            args=(gauss_seeds, gauss_coeffs, s_axis, which, mom_beta_arg, tilt_params))[0]

        update_l2 = math.sqrt(max(res["delta_sq"], 0.0))
        signed_disp = res["signed_disp"]
        disp_frac = (signed_disp / res["axis_norm"]) if res["axis_norm"] > 0 else 0.0
        cum_disp += signed_disp
        cum_disp_frac += disp_frac
        probe_pert_norm = (np.mean([av for av, _ in anchors]) * scale * axis_norm0) if anchors else 0.0
        scale_ratio = (sigma_scale / probe_pert_norm) if probe_pert_norm > 0 else float("inf")

        row = {
            "step": step, "warmup": step < warmup, "variant": a.variant, "axis": which,
            "population_size": N, "use_probe": use_probe,
            "axis_norm": axis_norm0, "axis_scale": scale,
            "signed_disp": signed_disp, "cum_disp": cum_disp,
            "signed_disp_frac": disp_frac, "cum_disp_frac": cum_disp_frac,
            "anchor_radii": [av for av, sg in anchors if sg > 0],
            "pair_fdiff": pair_fdiff, "s_axis": s_axis,
            "sigma": sigma, "sigma_ratio": sigma_ratio,
            "fs_ema": fs_ema, "fs_ref": fs_ref,
            # alpha is untouched by the controller, but the update is Delta = alpha*sigma*g_hat,
            # so the EFFECTIVE learning rate moves with sigma. Logged to keep that visible.
            "effective_lr": alpha * sigma,
            "sigma_scale": sigma_scale, "probe_pert_norm": probe_pert_norm,
            "scale_ratio": scale_ratio, "scale_flag": bool(scale_ratio > 10 or scale_ratio < 0.1),
            "fitness": [float(x) for x in f], "z": [float(x) for x in z],
            "correct_bits": member_bits, "batch_size": B,
            "gauss_seeds": gauss_seeds,
            "fitness_std": float(f.std()), "zero_update": bool(f.std() < 1e-12),
            "update_l2": update_l2, "tokens": step_tokens,
        }
        # Tilt-specific logging
        if tilt_variant:
            row["tilt_kappa"] = kappa
            row["tilt_kappa0"] = a.tilt_kappa
            row["tilt_kappa_end"] = a.tilt_kappa_end
            row["tilt_lambda"] = lam
            row["tilt_lambda_sched"] = lam_sched
            row["tilt_lambda_cosine"] = bool(a.tilt_lambda_cosine)
            row["tilt_half_vanilla"] = bool(a.tilt_half_vanilla)
            row["tilt_switch_frac"] = a.tilt_switch_frac if a.tilt_half_vanilla else None
            row["tilt_active"] = use_tilt
            row["mom_norm"] = mom_norm
            row["n_eff"] = n_eff
        # ---- advance the controller AFTER this step's fitness exists (causality) ----
        if a.sigma_adapt:
            fs = float(f.std())
            fs_ema = fs if fs_ema is None else a.sigma_adapt_ema * fs_ema + (1 - a.sigma_adapt_ema) * fs
            if step < a.sigma_adapt_warmup:
                fs_warm.append(fs)
            elif fs_ref is None:
                fs_ref = float(np.mean(fs_warm)) if fs_warm else None

        jf.write(json.dumps(row) + "\n"); jf.flush()
        if step % 5 == 0 or step < warmup:
            fd = ",".join(f"{x:+.3f}" for x in pair_fdiff) if pair_fdiff else "-"
            print(f"[{a.variant} N={N} s{a.pop_seed}] step {step} axis={axis_norm0:.3g} "
                  f"disp={signed_disp:+.3f} frac={disp_frac:+.4f} cum={cum_disp:+.2f} "
                  f"f+-f-=[{fd}] probe={use_probe} |D|={update_l2:.2e}", flush=True)

        # ---- intermediate evaluation every N updates ----
        # Runs AFTER commit, so we evaluate the committed θ_t, not a perturbed member.
        # axis_worker.es_commit_update_axis already updated _es_base and copied to model params,
        # so llm.generate() will use the correct weights.
        #
        # Gated on step+1, not step (changed 2026-08-28). step is 0-based and step 0 commits a
        # real update, so the commit above is update number step+1. The old `step % interval`
        # form therefore evaluated updates 11/21/.../91 while labelling them 10/20/.../90, and
        # never reached num_steps at all -- the 3B countdown run's last point is 91 of 100
        # updates with no measurement of the end state (COUNTDOWN_3B_SINGLE_GPU.md). Gating on
        # step+1 puts the curve on exactly interval/2*interval/.../num_steps updates, which
        # requires interval to divide num_steps. The old `step > 0` guard is gone: it is
        # redundant here and at interval=1 it wrongly dropped the first update.
        if a.eval_interval and (step + 1) % a.eval_interval == 0:
            import eval_core
            n_upd = step + 1
            print(f"[eval] intermediate eval after {n_upd} updates...", flush=True)
            t_eval = time.perf_counter()
            eval_result = eval_core.eval_on_llm(
                llm, tok, cap=a.eval_cap, max_tokens=max_tokens,
                perq_prefix=f"{a.out_prefix}_eval_upd{n_upd:03d}",
                include_countdown=(dataset == "countdown")
            )
            eval_sec = time.perf_counter() - t_eval
            # Both keys: `updates` is the meaningful axis, `step` keeps rows joinable to the
            # per-step jsonl. Curves written before this change carry `step` only, and their
            # step k is k+1 updates -- do not overlay the two without relabelling.
            rec = {"updates": n_upd, "step": step, "eval_result": eval_result, "eval_seconds": eval_sec}
            eval_history.append(rec)
            if ef is not None:
                ef.write(json.dumps(rec) + "\n"); ef.flush()
            # Print a one-line summary
            cd_acc = eval_result.get("countdown", {}).get("accuracy") if isinstance(eval_result.get("countdown"), dict) else None
            print(f"[eval] {n_upd} updates done in {eval_sec:.1f}s — countdown acc={cd_acc:.4f}" if cd_acc else
                  f"[eval] {n_upd} updates done in {eval_sec:.1f}s", flush=True)

    jf.close()
    if ef is not None:
        ef.close()

    kl_drift = None
    if a.kl and kl_rec is not None:
        import kl as klmod
        kl_drift = klmod.drift(llm, tok, kl_rec)
        print(f"[kl] final drift D={kl_drift:.4f}", flush=True)

    _peak = llm.collective_rpc("es_max_mem")[0]
    rows = [json.loads(l) for l in open(a.out_prefix + ".jsonl")]
    timed = [r for r in rows if not r["warmup"]]
    n_tie = sum(1 for r in timed for x in r["pair_fdiff"] if x == 0.0)
    n_pair = sum(len(r["pair_fdiff"]) for r in timed)
    summary = {
        "variant": a.variant, "axis": which, "model": C.MODEL,
        "population_size": N, "pop_seed": a.pop_seed, "num_steps": num_steps,
        "continuous_reward": a.continuous_reward if dataset == "countdown" else None,
        "a_max": a.a_max, "anchor_threshold": a.anchor_threshold,
        "mom_beta": a.mom_beta if which == "mom" else None,
        "mom_warmup": a.mom_warmup if which == "mom" else None,
        "disp_sign_means": {"base": "+ toward theta0", "mom": "+ along recent trajectory"}.get(which),
        "final_axis_norm": rows[-1]["axis_norm"] if rows else 0.0,
        "final_cum_disp": cum_disp,
        "final_cum_disp_frac": cum_disp_frac,
        "mean_signed_disp": float(np.mean([r["signed_disp"] for r in timed])) if timed else 0.0,
        "mean_signed_disp_frac": float(np.mean([r["signed_disp_frac"] for r in timed])) if timed else 0.0,
        "mean_pair_fdiff": float(np.mean([x for r in timed for x in r["pair_fdiff"]])) if n_pair else None,
        "pair_tie_rate": (n_tie / n_pair) if n_pair else None,
        "n_pairs_evaluated": n_pair,
        "steps_probe_active": int(sum(r["use_probe"] for r in timed)),
        "steps_scale_flagged": int(sum(r["scale_flag"] for r in timed)),
        "zero_update_rate": float(np.mean([r["zero_update"] for r in timed])) if timed else 0.0,
        "sigma_adapt": a.sigma_adapt,
        "sigma_adapt_gain": a.sigma_adapt_gain if a.sigma_adapt else None,
        "sigma_adapt_clip": a.sigma_adapt_clip if a.sigma_adapt else None,
        "sigma0": sigma0,
        "sigma_final": sigma,
        "sigma_ratio_mean": float(np.mean([r["sigma_ratio"] for r in timed])) if timed else 1.0,
        "sigma_ratio_min": float(np.min([r["sigma_ratio"] for r in timed])) if timed else 1.0,
        "sigma_ratio_max": float(np.max([r["sigma_ratio"] for r in timed])) if timed else 1.0,
        "gpu_mem_util": a.gpu_mem_util,
        "peak_mem_alloc_gb": _peak["max_allocated_bytes"] / 1e9,
        "peak_mem_reserved_gb": _peak["max_reserved_bytes"] / 1e9,
        "kl_proxy_drift": kl_drift,
        "s_per_step_mean": (time.perf_counter() - t_run0) / num_steps,
        "wall_clock_s": time.perf_counter() - t_run0,
        "eval_interval": a.eval_interval,
        "eval_history": eval_history,  # list of {step, eval_result, eval_seconds}
    }
    if tilt_variant:
        summary.update({
            "tilt_kappa": a.tilt_kappa,
            "tilt_kappa_end": a.tilt_kappa_end,
            "tilt_lambda_cosine": bool(a.tilt_lambda_cosine),
            "tilt_half_vanilla": bool(a.tilt_half_vanilla),
            "tilt_switch_frac": a.tilt_switch_frac if a.tilt_half_vanilla else None,
            "tilt_switch_step": (int(num_steps * a.tilt_switch_frac)
                                 if a.tilt_half_vanilla else None),
            "tilt_mom_beta": a.tilt_mom_beta,
            "tilt_warmup": a.tilt_warmup,
            "tilt_lambda0": kappa_to_lambda(a.tilt_kappa, d),
            "tilt_lambda_end": kappa_to_lambda(a.tilt_kappa_end, d),
        })
    if a.eval_final:
        import eval_core
        t0 = time.perf_counter()
        # A countdown-trained arm is measured on countdown (ID) AND the math battery (OOD),
        # the same direction as the 2026-07-30 battery. Adding countdown to a math-trained arm
        # would also be informative but changes what every existing math summary contains,
        # so it stays opt-in via the trained task.
        summary["eval_final"] = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=max_tokens,
                                                      perq_prefix=a.out_prefix + "_finaleval",
                                                      include_countdown=(dataset == "countdown"))
        summary["eval_max_tokens"] = max_tokens
        summary["eval_seconds"] = time.perf_counter() - t0
    with open(a.out_prefix + "_summary.json", "w") as f2:
        json.dump(summary, f2, indent=2)
    print("SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "eval_final"}, indent=2), flush=True)
    if "eval_final" in summary:
        print("EVAL_FINAL:", json.dumps(summary["eval_final"], indent=2), flush=True)


if __name__ == "__main__":
    main()
