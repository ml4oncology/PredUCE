# Predict Undesirable Cancer Events

Pipelines for data preparation, model development, and model evaluation for prediction of undesirable cancer events (i.e. symptom deterioration, emergency department visits, etc).

# Data Location
Please see [ml4oncology/make-clinical-dataset](https://github.com/ml4oncology/make-clinical-dataset).

# Getting started
```bash
git clone https://github.com/ml4oncology/PredUCE
pip install -e ".[dev]"

# optional
pre-commit install
nbstripout --install --keep-output
```

# Instructions
```bash
python <project folder>/scripts/run.py
```

# Project Organization
```
├── acu                <- Acute care use
│   ├── notebooks      <- Jupyter notebooks
│   ├── scripts        <- Python scripts
│   └── src            <- Python package where the main functionality goes
├── symp               <- Symptom deterioration
│   ├── notebooks
│   ├── scripts
│   └── src
├── shared             <- Shared modules between different projects
├── pyproject.toml     <- Build configuration
├── .env               <- Environment variables (i.e. personal keys)
```