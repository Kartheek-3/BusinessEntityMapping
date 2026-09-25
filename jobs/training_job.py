import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from src.config import FEATURE_COLUMNS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--model-dir", default="/opt/ml/model")
    args = parser.parse_args()

    df = pd.read_parquet(args.train)
    
    # Split by source1_entity_id
    splitter = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    
    groups = df["source1_entity_id"]
    if len(groups.unique()) > 1:
        train_idx, val_idx = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
    else:
        train_df = df
        val_df = df

    print(f"Training entities: {train_df['source1_entity_id'].nunique()}")
    print(f"Validation entities: {val_df['source1_entity_id'].nunique()}")
    print(f"Training pairs: {len(train_df)}")
    print(f"Validation pairs: {len(val_df)}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / positive if positive > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, "model.json"))
    
    # Generate validation predictions
    X_val = val_df[FEATURE_COLUMNS]
    val_probs = model.predict_proba(X_val)[:, 1]
    
    val_df = val_df.copy()
    val_df["match_probability"] = val_probs
    
    val_df.to_parquet(os.path.join(args.model_dir, "validation_predictions.parquet"), index=False)
    
    print("Training completed.")

if __name__ == "__main__":
    main()
