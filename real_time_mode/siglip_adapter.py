from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .smart_keyframes_and_classify import classify_frame_siglip, init_siglip_classifier


class SiglipScreenClassifier:
    def __init__(self) -> None:
        self._initialized = False
        self._model = None
        self._processor = None
        self._pack = None
        self._device = "cpu"

    def initialize(self) -> None:
        if self._initialized:
            return
        self._model, self._processor, self._pack, self._device = init_siglip_classifier()
        self._initialized = True

    def classify(self, frame_bgr: Any) -> Tuple[str, Dict[str, float]]:
        self.initialize()
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
