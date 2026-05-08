from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

from .state import ModeRouteDecision


def normalize_mode(label: str) -> str:
    compact = str(label or "").strip().lower()
    if compact in {"slide", "slides"}:
        return "slide"
    if compact in {"code", "demo", "other"}:
        return compact
    if compact in {"none", "no_share", "blank"}:
        return "other"
    return "other"


@dataclass
class ModeRouterState:
    current_mode: str = "other"
    previous_mode: str = "other"
    last_stable_mode: str = "other"
    pending_mode: Optional[str] = None
    pending_count: int = 0
    history: Deque[ModeRouteDecision] = field(default_factory=lambda: deque(maxlen=8))


class ModeRouter:
    def __init__(
        self,
        *,
        immediate_switch_confidence: float = 0.82,
        sticky_confidence_floor: float = 0.48,
        demo_guard_margin: float = 0.06,
        required_consecutive_frames: int = 2,
    ) -> None:
        self.immediate_switch_confidence = float(immediate_switch_confidence)
        self.sticky_confidence_floor = float(sticky_confidence_floor)
        self.demo_guard_margin = float(demo_guard_margin)
        self.required_consecutive_frames = int(required_consecutive_frames)
        self.state = ModeRouterState()

    def route(
        self,
        *,
        raw_label: str,
        raw_scores: Dict[str, float],
        demo_guard_active: bool = False,
    ) -> Tuple[ModeRouteDecision, bool]:
        proposed_mode = normalize_mode(raw_label)
        scores = _normalize_scores(raw_scores, fallback_mode=proposed_mode)
        if proposed_mode not in scores:
            scores.setdefault(proposed_mode, float(max(scores.values()) if scores else 0.0))
        confidence = float(scores.get(proposed_mode, 0.0))
        best_mode = max(scores, key=scores.get) if scores else proposed_mode
        best_confidence = float(scores.get(best_mode, confidence))
        previous_stable = self.state.last_stable_mode

        if demo_guard_active and best_mode != "demo":
            demo_score = float(scores.get("demo", 0.0))
            if demo_score >= max(best_confidence - self.demo_guard_margin, self.sticky_confidence_floor):
                best_mode = "demo"
                best_confidence = demo_score
                decision_path = "demo_guard_forced"
            else:
                decision_path = "best_score"
        else:
            decision_path = "best_score"

        mode_changed = False
        stable_mode = previous_stable
        if best_mode == previous_stable:
            self.state.pending_mode = None
            self.state.pending_count = 0
            stable_mode = best_mode
            decision_path = f"{decision_path}|stable_kept"
        elif best_confidence >= self.immediate_switch_confidence:
            stable_mode = best_mode
            mode_changed = True
            self.state.pending_mode = None
            self.state.pending_count = 0
            decision_path = f"{decision_path}|high_conf_switch"
        elif best_confidence < self.sticky_confidence_floor:
            stable_mode = previous_stable
            self.state.pending_mode = None
            self.state.pending_count = 0
            decision_path = f"{decision_path}|low_conf_keep_prev"
        else:
            if self.state.pending_mode == best_mode:
                self.state.pending_count += 1
            else:
                self.state.pending_mode = best_mode
                self.state.pending_count = 1
            if self.state.pending_count >= self.required_consecutive_frames:
                stable_mode = best_mode
                mode_changed = True
                self.state.pending_mode = None
                self.state.pending_count = 0
                decision_path = f"{decision_path}|consecutive_switch"
            else:
                stable_mode = previous_stable
                decision_path = f"{decision_path}|pending_confirm"

        decision = ModeRouteDecision(
            mode=stable_mode,
            confidence=float(scores.get(stable_mode, best_confidence)),
            decision_path=decision_path,
            scores=scores,
        )
        self.state.previous_mode = self.state.current_mode
        self.state.current_mode = stable_mode
        if mode_changed or not self.state.history:
            self.state.last_stable_mode = stable_mode
        self.state.history.append(decision)
        return decision, mode_changed


def _normalize_scores(raw_scores: Dict[str, float], *, fallback_mode: str) -> Dict[str, float]:
    scores = {
        "slide": 0.0,
        "code": 0.0,
        "demo": 0.0,
        "other": 0.0,
    }
    for key, value in dict(raw_scores or {}).items():
        normalized_key = normalize_mode(key)
        if str(key).startswith("_"):
            continue
        if normalized_key not in scores:
            continue
        scores[normalized_key] = max(scores.get(normalized_key, 0.0), float(value))
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 1e-6:
        best = normalize_mode(fallback_mode)
        scores[best] = 1.0
        return scores
    return {key: float(max(value, 0.0) / total) for key, value in scores.items()}
