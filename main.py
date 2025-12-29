# main.py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from medmnist import BreastMNIST
from torchvision import transforms
from torch.utils.data import DataLoader,random_split

from Code.model_A.features import extract_features
from Code.model_A.train_svm import train_model
from Code.model_A.evaluate import evaluate_model

from Code.model_B.cnn import SimpleCNN, DeepCNN
from Code.model_B.train_cnn import train_cnn, evaluate_cnn
from plot_results import plot_model_A_feature_pipeline,plot_training_budget_f1,plot_model_capacity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from complexity import get_model_complexity

DEV_MODE = False   # env False
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def run_model_A():
     # ===== Step 1: Flatten images =====
    DATA_ROOT = "Datasets"

    trainset = BreastMNIST(
        root=DATA_ROOT,
        split="train",
        download=DEV_MODE
    )
    testset = BreastMNIST(
        root=DATA_ROOT,
        split="test",
        download=DEV_MODE
    )

    X_train = trainset.imgs
    y_train = trainset.labels.flatten()
    X_test = testset.imgs
    y_test = testset.labels.flatten()

    X_train_raw = extract_features(X_train)
    X_test_raw  = extract_features(X_test)

    scaler = StandardScaler().fit(X_train_raw)
    X_train_raw = scaler.transform(X_train_raw)
    X_test_raw  = scaler.transform(X_test_raw)

    # ===== Raw features =====
    model_raw = train_model(X_train_raw, y_train, model_type="svm")
    results_raw = evaluate_model(model_raw, X_test_raw, y_test)


    # PCA
    pca = PCA(n_components=50).fit(X_train_raw)
    X_train_pca = pca.transform(X_train_raw)
    X_test_pca  = pca.transform(X_test_raw)
    model_pca = train_model(X_train_pca, y_train, model_type="svm")
    results_pca = evaluate_model(model_pca, X_test_pca, y_test)

    return {
        "raw": results_raw,
        "pca": results_pca
    }

def run_model_B():
    train_val_histories = {}
    # no augmentation (baseline)
    transform_plain = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    # with data augmentation
    transform_aug = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    DATA_ROOT = "Datasets"

    trainset_plain = BreastMNIST(
        root=DATA_ROOT,
        split="train",
        transform=transform_plain,
        download=DEV_MODE
    )

    trainset_aug = BreastMNIST(
        root=DATA_ROOT,
        split="train",
        transform=transform_aug,
        download=DEV_MODE
    )
    valset_plain = BreastMNIST(
        root=DATA_ROOT,
        split="val",
        transform=transform_plain,
        download=DEV_MODE
    )

    testset = BreastMNIST(
        root=DATA_ROOT,
        split="test",
        transform=transform_plain,
        download=DEV_MODE
    )

    print("train:", len(trainset_plain))
    print("val:  ", len(valset_plain))
    print("test: ", len(testset))
    total = len(trainset_plain) + len(valset_plain) + len(testset)
    print("ratios:",
      len(trainset_plain)/total,
      len(valset_plain)/total,
      len(testset)/total)

  
    train_loader_plain = DataLoader(trainset_plain, batch_size=64, shuffle=True)

    train_loader_aug = DataLoader(trainset_aug, batch_size=64, shuffle=True)

    test_loader = DataLoader(testset, batch_size=64)
    val_loader = DataLoader(  valset_plain,batch_size=64,shuffle=False)

    epoch_list = [10, 30, 50]
    f1_no_aug = []
    f1_aug = []

 
    for epochs in epoch_list:

        print(f"\nTraining SimpleCNN (no augmentation),epochs={epochs}")
        model_plain = SimpleCNN()
        history_plain = train_cnn(model_plain, train_loader_plain, val_loader, epochs=epochs)
        metrics_plain = evaluate_cnn(model_plain, test_loader)
        f1_no_aug.append(metrics_plain["f1"])

        print(f"\nTraining SimpleCNN (with augmentation),epochs={epochs}")
        model_aug = SimpleCNN()
        history_aug=train_cnn(model_aug, train_loader_aug, val_loader, epochs=epochs)
        metrics_aug = evaluate_cnn(model_aug, test_loader)
        f1_aug.append(metrics_aug["f1"])

        if epochs == 30:
            simple_best_metrics = metrics_plain
            train_val_histories["plain"] = history_plain
            train_val_histories["aug"] = history_aug
   
    print("\nTraining DeepCNN (30 epochs)")
    model_deep = DeepCNN()
    train_cnn(model_deep, train_loader_plain, val_loader, epochs=30)
    results_deep = evaluate_cnn(model_deep, test_loader)

    return {
        "epochs": epoch_list,
        "f1_no_aug": f1_no_aug,
        "f1_aug": f1_aug,
        "simple_best": f1_no_aug[1],  # 30 epochs
        "simple": simple_best_metrics,
        "deep": results_deep,
        "train_val_histories": train_val_histories
        }

def plot_train_val_loss(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Train–val loss figure saved to {save_path}")

def print_latex_table(results_A, results_B):
    print("\n% ===== Table: Overall Results =====")
    # SVM Raw
    r = results_A["raw"]
    print(
        f"SVM (Raw) & {r['accuracy']:.3f} & {r['precision']:.3f} & "
        f"{r['recall']:.3f} & {r['f1']:.3f} \\\\"
    )

    # SVM PCA
    r = results_A["pca"]
    print(
        f"SVM (PCA) & {r['accuracy']:.3f} & {r['precision']:.3f} & "
        f"{r['recall']:.3f} & {r['f1']:.3f} \\\\"
    )

    # SimpleCNN
    r = results_B["simple"]
    print(
        f"SimpleCNN & {r['accuracy']:.3f} & {r['precision']:.3f} & "
        f"{r['recall']:.3f} & {r['f1']:.3f} \\\\"
    )

    # DeepCNN
    r = results_B["deep"]
    print(
        f"DeepCNN & {r['accuracy']:.3f} & {r['precision']:.3f} & "
        f"{r['recall']:.3f} & {r['f1']:.3f} \\\\"
    )

def run_model_complexity_analysis():
    print("\nRunning Model Complexity Analysis...")
    print("----------------------------------")

    complexity = get_model_complexity()

    for model_name, param_count in complexity.items():
        print(f"{model_name} parameters: {param_count:,}")

    return complexity

def plot_model_complexity(complexity_dict, save_path="fig_model_complexity.png"):
    """
    Plot model complexity as a bar chart.
    """
    model_names = list(complexity_dict.keys())
    param_counts = list(complexity_dict.values())

    plt.figure()
    plt.bar(model_names, param_counts)
    plt.ylabel("Number of trainable parameters")
    plt.title("Model Complexity Comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Model complexity figure saved to {save_path}")

if __name__ == "__main__":
    results_A= run_model_A()
    results_B= run_model_B()

    # --- Figure 1: SVM Raw vs PCA ---
    plot_model_A_feature_pipeline(
        labels=["Raw features", "PCA features"],
        f1_scores=[results_A["raw"]["f1"], results_A["pca"]["f1"]],
        save_path="fig_svm_raw_vs_pca.png"
    )

    # --- Figure 2: Model capacity ---
    plot_model_capacity(
        model_names=["Model A (SVM)", "SimpleCNN", "DeepCNN"],
        f1_scores=[
            results_A["pca"]["f1"],
            results_B["simple_best"],
            results_B["deep"]["f1"]
        ],
        save_path="fig_model_capacity_f1.png"
    )

    # --- Figure 3: Training budget ---
    plot_training_budget_f1(
        epochs=results_B["epochs"],
        f1_no_aug=results_B["f1_no_aug"],
        f1_aug=results_B["f1_aug"],
        save_path="fig_training_budget_f1.png"
    )

    print_latex_table(results_A, results_B)
    complexity_results = run_model_complexity_analysis()
    plot_model_complexity(complexity_results)

    histories = results_B["train_val_histories"]

    plot_train_val_loss(
        histories["plain"],
        save_path="fig_train_val_loss_plain.png"
    )

    plot_train_val_loss(
        histories["aug"],
        save_path="fig_train_val_loss_aug.png"
    )
