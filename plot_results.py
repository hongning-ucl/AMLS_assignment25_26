import matplotlib.pyplot as plt


# =========================================================
# Figure 1: Training budget vs F1 (SimpleCNN)
# =========================================================
def plot_training_budget_f1(
    epochs,
    f1_no_aug,
    f1_aug,
    save_path="fig_training_budget_f1.png"
):
    """
    Plot training budget (epochs) vs F1 score for SimpleCNN.
    """

    plt.figure()
    plt.plot(epochs, f1_no_aug, marker="o", label="No Augmentation")
    plt.plot(epochs, f1_aug, marker="o", label="With Augmentation")

    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training Budget vs F1 Score (SimpleCNN)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Figure 2: Model capacity comparison
# =========================================================
def plot_model_capacity(
    model_names,
    f1_scores,
    save_path="fig_model_capacity_f1.png"
):
    """
    Plot F1-score comparison across models with different capacities.
    """

    plt.figure()
    plt.bar(model_names, f1_scores)

    plt.ylabel("F1 Score")
    plt.title("Model Capacity Comparison (F1 Score)")
    plt.ylim(0.80, 0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Figure 3: Raw vs PCA feature comparison (Model A - SVM)
# =========================================================
def plot_model_A_feature_pipeline(
    labels,
    f1_scores,
    save_path="fig_svm_raw_vs_pca.png"
):
    """
    Plot feature pipeline comparison for Model A (SVM).
    """

    plt.figure()
    plt.bar(labels, f1_scores)

    plt.ylabel("F1 Score")
    plt.title("Model A (SVM): Raw vs PCA Feature Pipeline")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()