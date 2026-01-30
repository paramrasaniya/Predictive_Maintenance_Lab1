import os
import nbformat as nbf


def make_notebook(title: str, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"title": title}
    return nb


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main():
    os.makedirs("notebooks", exist_ok=True)

    # EDA
    eda_cells = [
        md("# EDA.ipynb\nQuick inspection of the robot dataset."),
        code(
            "import pandas as pd\n"
            "df = pd.read_csv('../data/raw/RMBR4-2_export_test.csv')\n"
            "df.head(), df.shape, df.columns"
        ),
        code(
            "df['Trait'].value_counts().head(10)"
        )
    ]
    nbf.write(make_notebook("EDA", eda_cells), "notebooks/EDA.ipynb")

    # Linear Regression
    lr_cells = [
        md("# linear_regression.ipynb\nManual LR vs scikit-learn comparison."),
        code(
            "import yaml\n"
            "from src.run_experiment import main\n"
            "main()"
        )
    ]
    nbf.write(make_notebook("Linear Regression", lr_cells), "notebooks/linear_regression.ipynb")

    # RobotPM_MLOps
    mlops_cells = [
        md("# RobotPM_MLOps.ipynb\n\n"
           "## What changed vs the original streaming workshop?\n"
           "- Added modular `src/` architecture (separation of concerns)\n"
           "- Added YAML config (`configs/experiment_config.yaml`) for parameterized runs\n"
           "- Added experiment tracking (`experiments/results.csv`)\n"
           "- Added reproducible processed dataset output (`data/processed/`)\n"
           "- Added dashboard app (`dashboard/app.py`) for live monitoring + alerting\n\n"
           "## Recommended Additions\n"
           "- CI checks (lint/test), model registry, data validation\n\n"
           "## Recommended Enhancements\n"
           "- Replace engineered failure labels with real maintenance logs\n"
           "- Add multivariate regression / regularization\n"
           "- Add drift detection + retraining triggers\n"),
        code(
            "import pandas as pd\n"
            "df = pd.read_csv('../data/processed/processed_robot_data.csv')\n"
            "df[['robot_id','Time','failure_event','time_to_failure_days']].tail(10)"
        )
    ]
    nbf.write(make_notebook("RobotPM_MLOps", mlops_cells), "notebooks/RobotPM_MLOps.ipynb")

    print("Notebooks generated under /notebooks")


if __name__ == '__main__':
    main()
