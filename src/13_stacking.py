# src/06_stacking.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb

from prepare_kfold import load_full_data, get_kfold


if __name__ == "__main__":
    (
        project_dir,
        feature_cols,
        num_features,
        cat_features,
        X,
        y,
    ) = load_full_data()

    n_splits = 5
    kf = get_kfold(n_splits=n_splits, random_state=42)

    n_samples = X.shape[0]

    # OOF-Predictions für jedes Base-Model
    oof_cat = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_lgb = np.zeros(n_samples)

    # Testpreds kannst du später ergänzen (z.B. Mittel über Fold-Models)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost
        if cat_features:
            cat_idx = [feature_cols.index(c) for c in cat_features]
        else:
            cat_idx = []

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx if cat_idx else None)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx if cat_idx else None)

        cat_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_state=42,
            task_type="CPU",
        )
        cat_model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )
        oof_cat[val_idx] = cat_model.predict(X_val)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="rmse",
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
        )
        oof_xgb[val_idx] = xgb_model.predict(X_val)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="regression",
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_metric="rmse",
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val)

    # Qualität der Base-Models (OOF)
    rmse_cat = np.sqrt(mean_squared_error(y, oof_cat))
    rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))

    print(f"\n[OOF] CatBoost RMSE:  {rmse_cat:.5f}")
    print(f"[OOF] XGBoost RMSE:   {rmse_xgb:.5f}")
    print(f"[OOF] LightGBM RMSE:  {rmse_lgb:.5f}")

    # Stacking-Matrix
    oof_stack = np.vstack([oof_cat, oof_xgb, oof_lgb]).T  # shape (n_samples, 3)

    # Meta-Modell (z.B. Ridge)
    meta = Ridge(alpha=1.0, random_state=42)
    meta.fit(oof_stack, y)

    oof_meta = meta.predict(oof_stack)
    rmse_meta = np.sqrt(mean_squared_error(y, oof_meta))
    r2_meta = r2_score(y, oof_meta)

    print(f"\n[OOF] Stacking RMSE: {rmse_meta:.5f}")
    print(f"[OOF] Stacking R2:   {r2_meta:.5f}")

