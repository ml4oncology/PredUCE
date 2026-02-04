"""Multimodal fusion dataset"""
import polars as pl
import torch
from torch.utils.data import Dataset


class EmbeddingStore:
    """Efficient embedding lookup store backed by a torch tensor."""

    def __init__(self, path: str):
        df = pl.read_parquet(path, columns=['text_id', 'embedding'])
        self._embeddings = df['embedding'].to_torch() # (N, emb_dim)
        self._id_to_idx = {tid: idx for idx, tid in enumerate(df['text_id'])} 

    def __getitem__(self, text_id: int) -> torch.Tensor:
        return self._embeddings[self._id_to_idx[text_id]]

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, text_id: int) -> bool:
        return text_id in self._id_to_idx
    
    @property
    def dim(self) -> int:
        return self._embeddings.shape[1]


class FusionDataset(Dataset):
    """Dataset for multimodal fusion model.

    Provides batches with:
    - Continuous tabular features
    - Categorical tabular features (whose values are indices for learned embeddings)
    - Text embeddings (concatenated from multiple sections)
    - Mask indicating which samples have text embeddings
    """

    def __init__(
        self,
        X_tabular: pl.DataFrame,
        X_embedding: pl.DataFrame,
        y: pl.DataFrame,
        categ_cols: list[str],
        embedding_store: EmbeddingStore,
    ):
        """
        Args:
            X_tabular: Tabular features (continuous + categorical)
            X_embedding: Text embedding IDs
            y: Target labels
            categ_cols: Categorical column names
            embedding_store: Embedding lookup store
        """
        # Separate continuous and categorical data
        cont_cols = [c for c in X_tabular.columns if c not in categ_cols]
        self.cont_feats = X_tabular.select(cont_cols).to_torch(dtype=pl.Float32)
        self.categ_feats = {col: X_tabular[col].to_torch() for col in categ_cols}
        self.targs = y.to_torch(dtype=pl.Int8)

        # Text embedding lookup
        self.text_ids = X_embedding.to_numpy()  # (N, num_sections)
        self.num_sections = len(X_embedding.columns)
        self.emb_store = embedding_store


    def __len__(self) -> int:
        return len(self.targs)
    

    def __getitem__(self, idx: int) -> dict:
        # Continuous features
        cont = self.cont_feats[idx]

        # Categorical features
        categ = {col: arr[idx] for col, arr in self.categ_feats.items()}

        # Concatenate embeddings from all text sections
        text_ids = self.text_ids[idx]
        zero_emb = torch.zeros(self.emb_store.dim, dtype=torch.float32)
        emb = torch.concat([
            zero_emb if tid is None else self.emb_store[tid] 
            for tid in text_ids
        ])
        has_emb = torch.tensor(any(tid is not None for tid in text_ids))

        # Target
        target = self.targs[idx]

        return {
            "tabular_cont_feats": cont,
            "tabular_categ_feats": categ,
            "embedding_feats": emb,
            "has_embedding": has_emb,
            "target": target,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch samples into tensors.
    
    Used in the torch.utils.data.DataLoader.
    i.e. loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    # Stack categorical features by column
    categ_cols = batch[0]["tabular_categ_feats"].keys()
    categ = {
        col: torch.stack([b["tabular_categ_feats"][col] for b in batch]).cuda()
        for col in categ_cols
    }

    cont = torch.stack([b["tabular_cont_feats"] for b in batch]).cuda()
    emb = torch.stack([b["embedding_feats"] for b in batch]).cuda()
    has_emb = torch.stack([b["has_embedding"] for b in batch]).cuda()
    target = torch.stack([b["target"] for b in batch]).cuda()

    return {
        "tabular_cont_feats": cont,
        "tabular_categ_feats": categ,
        "embedding_feats": emb,
        "has_embedding": has_emb,
        "target": target,
    }