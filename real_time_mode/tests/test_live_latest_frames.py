from __future__ import annotations

from real_time_mode.live_app import PipelineSessionManager, ProcessFrameRequest, _stale_frame_response


def test_session_manager_tracks_latest_client_frame() -> None:
    manager = PipelineSessionManager()

    fresh, latest = manager.note_client_frame("session-a", 1)
    assert fresh is True
    assert latest == 1

    fresh, latest = manager.note_client_frame("session-a", 3)
    assert fresh is True
    assert latest == 3

    fresh, latest = manager.note_client_frame("session-a", 2)
    assert fresh is False
    assert latest == 3


def test_stale_frame_response_is_lightweight_and_marked_dropped() -> None:
    payload = ProcessFrameRequest(
        session_id="session-a",
        timestamp=12.5,
        client_frame_id=2,
        client_sent_at=10.0,
        client_mode_hint="demo",
        frame_jpeg_base64="data:image/jpeg;base64,ignored",
    )

    response = _stale_frame_response("session-a", payload, latest_frame_id=4)

    assert response["stale"] is True
    assert response["dropped"] is True
    assert response["client_frame_id"] == 2
    assert response["latest_client_frame_id"] == 4
    assert response["processor_used"] == "stale_drop"
    assert response["raw_record"] == {}
