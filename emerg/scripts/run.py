import os
import json
import datetime
from dotenv import load_dotenv

from make_clinical_dataset.shared.constants import ROOT_DIR
from preduce.emerg.config import ModelConfig, TrainConfig
from preduce.emerg.dataset import EmbeddingStore, FusionDataset, collate_fn
from preduce.emerg.model import FusionModel
from preduce.emerg.prep import build_features, load_data, prepare_splits
from preduce.emerg.train import Trainer
from torch.utils.data import DataLoader

load_dotenv()

# TODO: move to .env file
EMB_MODEL = "PubMedBERT"
SPLIT_DATE = "2022-01-01"
TARGETS = ["target_ED_30d", "target_ED_60d", "target_ED_90d"]


DATE = "2025-03-29"
DATA_DIR = f"{ROOT_DIR}/data/final/data_{DATE}"
DATA_PATH = f"{DATA_DIR}/processed/treatment_centered_data.parquet"
DATE_PATH = f"{DATA_DIR}/processed/treatment_centered_dates.parquet"
NOTE_PATH = f"{DATA_DIR}/interim/subsets/clinic_visits_prior_to_treatment/notes.parquet"

TEXT_PATH = f"{DATA_DIR}/interim/embedding/ed_risk_summary.parquet"
EMB_PATH = f"{DATA_DIR}/interim/embedding/{EMB_MODEL}"

SAVE_DIR = f"{os.getenv('SAVE_DIR')}/experiment_001"
os.makedirs(SAVE_DIR, exist_ok=True)


def main(first_visit_only: bool):
    df = load_data(DATA_PATH, NOTE_PATH, TEXT_PATH)
    df = build_features(df)
    data = prepare_splits(df, split_date=SPLIT_DATE)

    # Set up config
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    categ_sizes = {col: df[col].n_unique() for col in ["primary_site_desc_idx"]}

    # Set up embedding lookup table
    emb_store = EmbeddingStore(EMB_PATH)

    # Set up data loaders
    dataloaders = {}
    for split in ["train", "valid", "test"]:
        dataset = FusionDataset(
            data[split]["X_tabular"],
            data[split]["X_embedding"],
            data[split]["y"].select(TARGETS),
            categ_cols=list(categ_sizes),
            embedding_store=emb_store,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Set up the model
    model = FusionModel(
        tabular_input_dim=len(data["train"]["X_tabular"].columns),
        embedding_input_dim=emb_store.dim * len(data["train"]["X_embedding"].columns),
        categ_sizes=categ_sizes,
        model_config=model_cfg,
        num_tasks=len(TARGETS),
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        config=train_cfg,
        save_dir=SAVE_DIR,
    )

    # Train the model
    results = trainer.train()

    # Save training history
    with open(f"{SAVE_DIR}/history.json", "w") as f:
        json.dump(results["history"], f, indent=2)

    # Save config
    cfg = {
        "text_embed_model": EMB_MODEL,
        "split_date": SPLIT_DATE,
        "created_at": datetime.now().isoformat(),
        "note": "treatment centered ED 90d prediction",
    }
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
