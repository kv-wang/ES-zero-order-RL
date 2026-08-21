import pandas as pd
import matplotlib.pyplot as plt


def main():
    curves = pd.read_csv("results/training_curves.csv")
    ood = pd.read_csv("results/ood_eval_summary.csv")

    plt.figure()
    plt.plot(curves["step"], curves["train_reward"], label="train")
    plt.plot(curves["step"], curves["val_reward"], label="val")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("Train/Val Reward")
    plt.savefig("plots/train_val_reward_curve.png", bbox_inches="tight")

    plt.figure()
    for ds, sdf in ood.groupby("eval_dataset"):
        plt.plot(sdf["checkpoint_step"], sdf["pass_at_1"], marker="o", label=ds)
    plt.legend()
    plt.xlabel("checkpoint_step")
    plt.ylabel("pass@1")
    plt.title("OOD pass@1 by checkpoint")
    plt.savefig("plots/ood_accuracy_by_checkpoint.png", bbox_inches="tight")

    plt.figure()
    bar = ood.groupby("eval_dataset", as_index=False)["pass_at_1"].mean()
    plt.bar(bar["eval_dataset"], bar["pass_at_1"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("pass@1")
    plt.title("pass@1 by dataset")
    plt.savefig("plots/pass1_by_dataset_barplot.png", bbox_inches="tight")

    if "population_size" in ood.columns:
        plt.figure()
        ab = ood.groupby("population_size", as_index=False)["pass_at_1"].mean()
        plt.plot(ab["population_size"], ab["pass_at_1"], marker="o")
        plt.xlabel("population_size")
        plt.ylabel("pass@1")
        plt.title("Population size ablation")
        plt.savefig("plots/population_size_ablation.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
