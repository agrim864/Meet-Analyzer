from .state import (
    ChangeHypothesis,
    CursorState,
    ModeRouteDecision,
    Overlay,
    SceneStateRecord,
    ScreenState,
    StateDelta,
    TranscriptSegment,
    UIElement,
)

__all__ = [
    "DemoAnalyzerPipeline",
    "IncrementalUIStatePipeline",
    "BatchMeetingAnalyzer",
    "ChangeHypothesis",
    "CursorState",
    "ModeRouteDecision",
    "Overlay",
    "SceneStateRecord",
    "ScreenState",
    "StateDelta",
    "TranscriptSegment",
    "UIElement",
]


def __getattr__(name: str):
    if name in {"DemoAnalyzerPipeline", "IncrementalUIStatePipeline"}:
        from .pipeline import DemoAnalyzerPipeline, IncrementalUIStatePipeline

        return {
            "DemoAnalyzerPipeline": DemoAnalyzerPipeline,
            "IncrementalUIStatePipeline": IncrementalUIStatePipeline,
        }[name]
    if name == "BatchMeetingAnalyzer":
        from .service import BatchMeetingAnalyzer

        return BatchMeetingAnalyzer
    raise AttributeError(f"module 'batch_mode' has no attribute {name!r}")
