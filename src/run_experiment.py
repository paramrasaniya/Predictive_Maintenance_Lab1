from __future__ import annotations

import os
import csv
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR

from src.data_loader import load_csv, add_robot_sources, DBConfig, get_engine, init_db_table, load_df_to_db, load_from_db
from src.preprocessing import parse_time, handle_missing, engineer_failure_and_target, select_univariate_xy
from src.model import LinearRegressionScratch
from src.evaluation import rmse, mae, r2_score, save_scatter_with_line, save_residual_plot


def _ensure_results_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_tag", "model_type", "feature", "target,",
                "rmse", "mae", "r2",
                "scratch_learning_rate", "scratch_epochs",
                "notes"
            ])


def main():
    cfg_path = "configs/experiment_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_tag = cfg["project"]["run_tag"]
    csv_path = cfg["data"]["csv_path"]
    create_three_robots = bool(cfg["data"]["create_three_robots"])
    robot_count = int(cfg["data"]["robot_count"])

    use_db = bool(cfg["data"]["use_db"])
    db_conn = cfg["data"]["db_connection_string"]
    db_table = cfg["data"]["db_table"]

    feature_col = cfg["features"]["predictor_feature"]
    z_threshold = float(cfg["targets"]["z_threshold"])
    min_gap_minutes = int(cfg["targets"]["min_gap_minutes_between_failures"])
    target_col = cfg["targets"]["target_name"]

    test_size = float(cfg["split"]["test_size"])
    random_state = int(cfg["split"]["random_state"])

    lr = float(cfg["scratch_model"]["learning_rate"])
    epochs = int(cfg["scratch_model"]["epochs"])

    processed_csv_path = cfg["outputs"]["processed_csv_path"]
    results_csv_path = cfg["outputs"]["results_csv_path"]
    plots_dir = cfg["outputs"]["plots_dir"]

    # 1) Load
    if use_db:
        engine = get_engine(DBConfig(connection_string=db_conn, table=db_table))
        # If table empty/nonexistent, seed it from CSV
        df_seed = load_csv(csv_path)
        if create_three_robots:
            df_seed = add_robot_sources(df_seed, robot_count=robot_count)
        df_seed = parse_time(df_seed)
        df_seed = handle_missing(df_seed)
        init_db_table(engine, db_table, df_seed)
        load_df_to_db(engine, db_table, df_seed)
        df = load_from_db(engine, db_table)
    else:
        df = load_csv(csv_path)
        if create_three_robots:
            df = add_robot_sources(df, robot_count=robot_count)

    # 2) Preprocess + feature engineering
    df = parse_time(df)
    df = handle_missing(df)

    if "robot_id" not in df.columns:
        df["robot_id"] = "robot_1"

    df = engineer_failure_and_target(
        df=df,
        robot_id_col="robot_id",
        time_col="Time",
        feature_col=feature_col,
        z_threshold=z_threshold,
        min_gap_minutes_between_failures=min_gap_minutes,
        target_name=target_col,
    )

    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    df.to_csv(processed_csv_path, index=False)

    # 3) Select X, y (univariate)
    # Keep one Trait as a clean slice (optional); many rows show Trait="current"
    # If your file contains multiple traits, you can filter here.
    X, y = select_univariate_xy(df, feature_col=feature_col, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4) Train scratch
    scratch = LinearRegressionScratch(learning_rate=lr, epochs=epochs).fit(X_train, y_train)
    y_pred_scratch = scratch.predict(X_test)

    # 5) Train sklearn
    sk = SklearnLR().fit(X_train, y_train)
    y_pred_sk = sk.predict(X_test)

    # 6) Evaluate
    scratch_rmse = rmse(y_test, y_pred_scratch)
    scratch_mae = mae(y_test, y_pred_scratch)
    scratch_r2 = r2_score(y_test, y_pred_scratch)

    sk_rmse = rmse(y_test, y_pred_sk)
    sk_mae = mae(y_test, y_pred_sk)
    sk_r2 = r2_score(y_test, y_pred_sk)

    # 7) Save plots
    os.makedirs(plots_dir, exist_ok=True)
    save_scatter_with_line(
        X_test, y_test, y_pred_scratch,
        out_path=os.path.join(plots_dir, f"{run_tag}_scratch_scatter_line.png"),
        title=f"{run_tag} | Scratch LR | y vs x"
    )
    save_residual_plot(
        y_test, y_pred_scratch,
        out_path=os.path.join(plots_dir, f"{run_tag}_scratch_residuals.png"),
        title=f"{run_tag} | Scratch LR | residuals"
    )
    save_scatter_with_line(
        X_test, y_test, y_pred_sk,
        out_path=os.path.join(plots_dir, f"{run_tag}_sklearn_scatter_line.png"),
        title=f"{run_tag} | Sklearn LR | y vs x"
    )
    save_residual_plot(
        y_test, y_pred_sk,
        out_path=os.path.join(plots_dir, f"{run_tag}_sklearn_residuals.png"),
        title=f"{run_tag} | Sklearn LR | residuals"
    )

    # 8) Track experiment results
    _ensure_results_csv(results_csv_path)
    with open(results_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            run_tag, "scratch", feature_col, target_col,
            f"{scratch_rmse:.6f}", f"{scratch_mae:.6f}", f"{scratch_r2:.6f}",
            lr, epochs,
            "Engineered failure target via rolling z-score"
        ])
        writer.writerow([
            run_tag, "sklearn", feature_col, target_col,
            f"{sk_rmse:.6f}", f"{sk_mae:.6f}", f"{sk_r2:.6f}",
            "", "",
            "Baseline sklearn LinearRegression"
        ])

    # 9) Print a clean summary
    print("\n=== RUN COMPLETE ===")
    print(f"Processed data saved: {processed_csv_path}")
    print(f"Results appended to:  {results_csv_path}")
    print(f"Plots saved in:       {plots_dir}\n")

    print("--- Scratch Model ---")
    print(f"RMSE: {scratch_rmse:.6f} | MAE: {scratch_mae:.6f} | R2: {scratch_r2:.6f}")
    print(f"w: {scratch.w:.6f} | b: {scratch.b:.6f}")

    print("\n--- Sklearn Model ---")
    print(f"RMSE: {sk_rmse:.6f} | MAE: {sk_mae:.6f} | R2: {sk_r2:.6f}")
    print(f"coef: {float(sk.coef_[0]):.6f} | intercept: {float(sk.intercept_):.6f}\n")


if __name__ == "__main__":
    main()
