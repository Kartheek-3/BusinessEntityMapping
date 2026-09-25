import argparse
import json
import os
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta = 0.5
    f05 = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall) if precision + recall else 0.0
    return precision, recall, f05

def calculate_entity_metrics(df, threshold):
    f05_scores = []
    
    for s1_id, group in df.groupby("source1_entity_id"):
        true_ids = set(group[group["label"] == 1]["candidate_entity_id"])
        pred_ids = set(group[group["match_probability"] >= threshold]["candidate_entity_id"])
        
        if not true_ids and not pred_ids:
            f05_scores.append(1.0)
            continue
            
        tp = len(true_ids & pred_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        beta = 0.5
        if precision + recall > 0:
            f05 = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
        else:
            f05 = 0.0
            
        f05_scores.append(f05)
        
    if not f05_scores:
        return 0.0
    return sum(f05_scores) / len(f05_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)

    best = None

    for threshold in np.arange(0.50, 0.96, 0.01):
        y_true = df["label"].values
        y_pred = (df["match_probability"].values >= threshold).astype(int)

        p, r, f = calculate_metrics(y_true, y_pred)
        entity_macro_f05 = calculate_entity_metrics(df, threshold)

        if best is None or f > best["f05"]:
            best = {
                "threshold": float(round(threshold, 2)),
                "precision": float(p),
                "recall": float(r),
                "f05": float(f),
                "entity_macro_f05": float(entity_macro_f05)
            }

    print("Best threshold:", round(best["threshold"], 4))
    print("Precision:", round(best["precision"], 4))
    print("Recall:", round(best["recall"], 4))
    print("F0.5:", round(best["f05"], 4))
    print("Entity Macro F0.5:", round(best["entity_macro_f05"], 4))
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "threshold.json"), "w") as f:
        json.dump(best, f, indent=4)

if __name__ == "__main__":
    main()
