from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from .smart_keyframes_and_classify import classify_frame_siglip, init_siglip_classifier

logger = logging.getLogger(__name__)


class SiglipScreenClassifier:
    def __init__(self) -> None:
        self._initialized = False
        self._model = None
        self._processor = None
        self._pack = None
        self._device = "cpu"
        self._fallback = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if os.getenv("BATCH_DISABLE_SIGLIP", "").strip().lower() in {"1", "true", "yes"}:
            logger.info("SigLIP disabled by BATCH_DISABLE_SIGLIP; using OCR route guards only.")
            self._model = None
            self._processor = None
            self._pack = None
            self._device = "fallback"
            self._fallback = True
            self._initialized = True
            return
        try:
            self._model, self._processor, self._pack, self._device = init_siglip_classifier()
        except Exception as exc:
            logger.warning("SigLIP initialization failed; using OCR route guards only: %s", exc)
            self._model = None
            self._processor = None
            self._pack = None
            self._device = "fallback"
            self._fallback = True
        self._initialized = True

    def classify(self, frame_bgr: Any) -> Tuple[str, Dict[str, float]]:
        self.initialize()
        if self._fallback:
            return "other", {"slide": 0.25, "code": 0.25, "demo": 0.25, "other": 0.25}
        label, scores = classify_frame_siglip(
            frame_bgr=frame_bgr,
            siglip_model=self._model,
            siglip_processor=self._processor,
            siglip_text_features=self._pack,
            siglip_device=self._device,
        )
        return str(label or "none").lower(), {str(k): float(v) for k, v in dict(scores or {}).items()}

    @property
    def device(self) -> str:
        return self._device
