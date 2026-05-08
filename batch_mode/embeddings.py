from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from packaging.version import Version

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    provider: str


class BgeM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_attempted = False

    def warmup(self) -> None:
        self._ensure_model()

    def encode(self, texts: Sequence[str]) -> EmbeddingResult:
        clean = [str(text or "").strip() for text in texts]
        if not clean:
            return EmbeddingResult(vectors=np.zeros((0, 8), dtype=np.float32), provider="empty")
        model = self._ensure_model()
        if model is None:
            return EmbeddingResult(vectors=_lexical_embeddings(clean), provider="lexical-fallback")
        vectors = model.encode(clean, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return EmbeddingResult(vectors=np.asarray(vectors, dtype=np.float32), provider=f"sentence-transformers:{self.model_name}")

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None
        self._load_attempted = True
        if SentenceTransformer is None:
            logger.warning("sentence-transformers unavailable; using lexical embedding fallback.")
            return None
        try:
            model_kwargs = {}
            try:
                import torch

                torch_version = Version(str(torch.__version__).split("+", 1)[0])
                if torch_version < Version("2.6.0"):
                    model_kwargs["use_safetensors"] = True
            except Exception:
                model_kwargs["use_safetensors"] = True
            if _huggingface_offline():
                model_kwargs["local_files_only"] = True

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
            )
        except Exception:
            logger.exception("Failed to load embedding model %s; falling back to lexical embeddings.", self.model_name)
            self._model = None
        return self._model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(item, dtype=np.float32) for item in vectors if np.asarray(item).size]
    if not valid:
        return np.zeros((8,), dtype=np.float32)
    max_dim = max(item.shape[-1] for item in valid)
    padded = []
    for item in valid:
        if item.shape[-1] < max_dim:
            pad = np.zeros((max_dim - item.shape[-1],), dtype=np.float32)
            item = np.concatenate([item, pad], axis=0)
        padded.append(item)
    return np.mean(np.stack(padded, axis=0), axis=0)


def _lexical_embeddings(texts: Sequence[str]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for text in texts:
        lower = text.lower()
        vec = np.array(
            [
                len(lower),
                lower.count("upload"),
                lower.count("slide"),
                lower.count("code"),
                lower.count("demo"),
                lower.count("error"),
                lower.count("issue"),
                lower.count("action"),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        vectors.append(vec)
    return np.stack(vectors, axis=0)


def _huggingface_offline() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").strip() in {"1", "true", "TRUE"} or os.getenv("TRANSFORMERS_OFFLINE", "").strip() in {"1", "true", "TRUE"}
