import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.utils import get_config

# Load config and target column
config = get_config(os.path.join(os.getcwd(), "config.yaml"))
target_column = config["TARGET"]

def validate_pred(pred_df, save_dir):
    """
    Validate predictions and save metrics and confusion matrix.
    
    Args:
        pred_df (pd.DataFrame): DataFrame with columns [target_column, "pred_subscription"]
        save_dir (str): Directory to save confusion matrix image and metrics JSON
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save raw predictions for reference
    pred_df.to_csv(os.path.join(save_dir, "pred.csv"), index=False)

    # Ensure numeric 0/1 for sklearn metrics
    pred_df["pred_subscription"] = pred_df["pred_subscription"].map({"yes": 1, "no": 0})
    pred_df[target_column] = pred_df[target_column].map({"yes": 1, "no": 0})

    y_true = pred_df[target_column]
    y_pred = pred_df["pred_subscription"]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

    # Save metrics as JSON
    metrics_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved evaluation metrics to {metrics_path}")
    print(f"Saved confusion matrix image to {cm_path}")

    return metrics
