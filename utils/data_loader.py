"""
data_loader.py — Carga y cómputo de estadísticas con caché de Streamlit.
"""
import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ─── Constantes ────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Prótesis", "Sano", "Caries", "Otro"]
CLASS_COLORS = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8"]

TRAIN_PATH = "train-00000-of-00001.parquet"
TEST_PATH  = "test-00000-of-00001.parquet"


# ─── Helpers ───────────────────────────────────────────────────────────────
def bytes_to_pil(raw) -> Image.Image:
    """Convierte bytes (dict o bytes) a imagen PIL RGB."""
    if isinstance(raw, dict):
        raw = raw.get("bytes", raw)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _stats_for_row(row) -> dict | None:
    try:
        img  = bytes_to_pil(row["image"])
        arr  = np.array(img, dtype=np.float32)
        gray = arr.mean(axis=2)
        raw  = row["image"]
        raw_bytes = raw.get("bytes", raw) if isinstance(raw, dict) else raw
        return {
            "width":      img.width,
            "height":     img.height,
            "aspect":     round(img.width / img.height, 3),
            "bytes_size": len(raw_bytes),
            "brightness": round(float(gray.mean()), 3),
            "contrast":   round(float(gray.std()),  3),
            "mean_r":     round(float(arr[:, :, 0].mean()), 3),
            "mean_g":     round(float(arr[:, :, 1].mean()), 3),
            "mean_b":     round(float(arr[:, :, 2].mean()), 3),
            "std_r":      round(float(arr[:, :, 0].std()),  3),
            "std_g":      round(float(arr[:, :, 1].std()),  3),
            "std_b":      round(float(arr[:, :, 2].std()),  3),
            "md5":        hashlib.md5(raw_bytes).hexdigest(),
        }
    except Exception:
        return None


# ─── Funciones cacheadas ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datasets…")
def load_dataframes():
    """Carga train y test desde parquet."""
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test  = pd.read_parquet(TEST_PATH)
    return df_train, df_test


@st.cache_data(show_spinner="Calculando estadísticas de imágenes (primera vez: ~3 min)…")
def compute_stats(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Extrae estadísticas por imagen. Se cachea para no recalcular."""
    def _extract(df):
        records = df.apply(_stats_for_row, axis=1).dropna()
        stats   = pd.DataFrame(list(records))
        if "label" in df.columns:
            stats["label"] = df["label"].values[: len(stats)]
        if "path"  in df.columns:
            stats["path"]  = df["path"].values[:  len(stats)]
        return stats

    return _extract(df_train), _extract(df_test)


@st.cache_data(show_spinner=False)
def get_sample_images(df: pd.DataFrame, cls: int, n: int = 5, seed: int = 42):
    """Devuelve n imágenes PIL de la clase dada."""
    subset = df[df["label"] == cls]
    sample = subset.sample(min(n, len(subset)), random_state=seed)
    images = []
    for _, row in sample.iterrows():
        try:
            images.append(bytes_to_pil(row["image"]))
        except Exception:
            pass
    return images
