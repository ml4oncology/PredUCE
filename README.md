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