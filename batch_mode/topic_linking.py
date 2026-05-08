from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from .embeddings import BgeM3Embedder, cosine_similarity, mean_vector


@dataclass
class TopicCluster:
    topic_id: str
    topic_title: str
    related_event_ids: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "related_event_ids": list(self.related_event_ids),
            "evidence": [dict(item) for item in self.evidence],
        }


class GlobalTopicLinker:
    def __init__(self, *, embedder: BgeM3Embedder | None = None, similarity_threshold: float = 0.69) -> None:
        self.embedder = embedder
        self.similarity_threshold = float(similarity_threshold)

    def cluster(self, records: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any] | Any]) -> List[TopicCluster]:
        candidates = self._build_candidates(records, units)
        if not candidates:
            return []
        vectors = self.embedder.encode([item["text"] for item in candidates]).vectors if self.embedder is not None else None
        clusters: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            placed = False
            vector = vectors[index] if vectors is not None else np.zeros((8,), dtype=np.float32)
            for cluster in clusters:
                similarity = cosine_similarity(cluster["vector"], vector) if vectors is not None else self._lexical_overlap(cluster["keywords"], candidate["keywords"])
                if similarity >= self.similarity_threshold:
                    cluster["items"].append(candidate)
                    cluster["vector"] = mean_vector([cluster["vector"], vector])
                    cluster["keywords"].update(candidate["keywords"])
                    placed = True
                    break
            if not placed:
                clusters.append({"items": [candidate], "vector": vector, "keywords": set(candidate["keywords"])})
        results: List[TopicCluster] = []
        for idx, cluster in enumerate(clusters, start=1):
            items = cluster["items"]
            title = self._cluster_title(items)
            event_ids: List[str] = []
            evidence: List[Dict[str, Any]] = []
            for item in items:
                event_ids.extend(item["event_ids"])
                evidence.append(
                    {
                        "time": item["time"],
                        "class": item["class"],
                        "text": item["text"][:320],
                    }
                )
            results.append(
                TopicCluster(
                    topic_id=f"topic_{idx:03d}",
                    topic_title=title,
                    related_event_ids=sorted(set(event_ids)),
                    evidence=evidence,
                )
            )
        return results

    def _build_candidates(self, records: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any] | Any]) -> List[Dict[str, Any]]:
        by_event = {str(item.get("event_id")): item for item in records}
        candidates: List[Dict[str, Any]] = []
        for unit in units:
            unit_dict = unit.to_dict() if hasattr(unit, "to_dict") else dict(unit)
            text = " ".join(
                part for part in [
                    str(unit_dict.get("unit_summary", "") or ""),
                    str(unit_dict.get("transcript_text", "") or ""),
                    str(unit_dict.get("ocr_text", "") or ""),
                ] if part
            ).strip()
            if not text:
                continue
            keywords = _keywords(text)
            candidates.append(
                {
                    "time": float(unit_dict.get("start", 0.0)),
                    "class": str(unit_dict.get("dominant_class", "other")),
                    "text": text,
                    "event_ids": [str(item) for item in unit_dict.get("event_ids", [])],
                    "keywords": keywords,
                }
            )
        for record in records:
            payload = record.get("visual_payload", {}) or {}
            text = " ".join(
                part for part in [
                    str(record.get("summary", "") or ""),
                    str(payload.get("page_summary", "") or ""),
                    str(payload.get("ocr_text", "") or ""),
                ] if part
            ).strip()
            if not text:
                continue
            candidates.append(
                {
                    "time": float(record.get("timestamp_sec", 0.0)),
                    "class": str(record.get("mode", "other")),
                    "text": text,
                    "event_ids": [str(record.get("event_id"))],
                    "keywords": _keywords(text),
                }
            )
        return candidates

    def _cluster_title(self, items: Sequence[Dict[str, Any]]) -> str:
        counter = Counter()
        for item in items:
            counter.update(item["keywords"])
        top = [word.replace("_", " ").title() for word, _ in counter.most_common(3)]
        return " / ".join(top) if top else "Recurring Topic"

    def _lexical_overlap(self, a: Sequence[str], b: Sequence[str]) -> float:
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        return float(len(set_a & set_b) / max(1, len(set_a | set_b)))


def _keywords(text: str) -> List[str]:
    tokens = [
        token.strip(".,:;!?()[]{}<>\"'").lower()
        for token in str(text or "").split()
    ]
    filtered = [token for token in tokens if len(token) >= 4 and token not in {"that", "this", "with", "from", "into", "mode"}]
    return filtered[:40]
