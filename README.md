# Predict Undesirable Cancer Events

Build models to predict undesirable cancer events (i.e. symptom deterioration, emergency department visits, etc).
Pipelines for data preparation, model development, and model evaluation. 

# Data Location
Please see [ml4oncology/make-clinical-dataset](https://github.com/ml4oncology/make-clinical-dataset).

# Getting started
```bash
git clone https://github.com/ml4oncology/PredUCE
conda env create -f envs/<env_name>.yaml
conda activate aim2reduce
pip install -e .

# optional
pre-commit install
```