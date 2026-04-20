"""Tests for Session CRUD operations."""

from __future__ import annotations

from app.crud.session import (
    count_by_user,
    create,
    get_by_id,
    list_by_user,
    soft_delete,
    update,
    update_session_analysis,
)
from app.models.session import Session, SessionMetric
from app.models.user import User


def _make_user(db, user_id: str = "user-1") -> User:
    user = User(id=user_id, email=f"{user_id}@test.com", hashed_password="hash")
    db.add(user)
    return user


async def test_create_session(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(
        db_session,
        user_id="user-1",
        element_type="waltz_jump",
        video_key="videos/test.mp4",
    )
    assert session.id is not None
    assert session.element_type == "waltz_jump"
    assert session.status == "uploading"
    assert session.video_key == "videos/test.mp4"


async def test_get_by_id(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    fetched = await get_by_id(db_session, session.id)
    assert fetched is not None
    assert fetched.element_type == "axel"


async def test_get_by_id_with_metrics(db_session):
    """Verify get_by_id works with sessions that have metrics."""
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    metric = SessionMetric(
        session_id=session.id,
        metric_name="airtime",
        metric_value=0.65,
    )
    db_session.add(metric)
    await db_session.flush()

    fetched = await get_by_id(db_session, session.id)
    assert fetched is not None
    assert fetched.element_type == "axel"


async def test_get_by_id_not_found(db_session):
    fetched = await get_by_id(db_session, "nonexistent")
    assert fetched is None


async def test_list_by_user_default_sort(db_session):
    _make_user(db_session)
    _make_user(db_session, "user-2")
    await db_session.flush()

    await create(db_session, user_id="user-1", element_type="axel")
    await create(db_session, user_id="user-1", element_type="waltz_jump")
    await create(db_session, user_id="user-2", element_type="axel")

    sessions = await list_by_user(db_session, "user-1")
    assert len(sessions) == 2


async def test_list_by_user_filter_element_type(db_session):
    _make_user(db_session)
    await db_session.flush()

    await create(db_session, user_id="user-1", element_type="axel")
    await create(db_session, user_id="user-1", element_type="waltz_jump")

    sessions = await list_by_user(db_session, "user-1", element_type="axel")
    assert len(sessions) == 1
    assert sessions[0].element_type == "axel"


async def test_list_by_user_sort_by_score(db_session):
    _make_user(db_session)
    await db_session.flush()

    s1 = await create(db_session, user_id="user-1", element_type="axel", overall_score=7.5)
    s2 = await create(db_session, user_id="user-1", element_type="axel", overall_score=9.0)

    sessions = await list_by_user(db_session, "user-1", sort="overall_score")
    assert len(sessions) == 2
    # Sorted desc by overall_score
    assert sessions[0].overall_score == 9.0
    assert sessions[1].overall_score == 7.5


async def test_list_by_user_pagination(db_session):
    _make_user(db_session)
    await db_session.flush()

    for _ in range(5):
        await create(db_session, user_id="user-1", element_type="axel")

    page1 = await list_by_user(db_session, "user-1", limit=2, offset=0)
    page2 = await list_by_user(db_session, "user-1", limit=2, offset=2)
    page3 = await list_by_user(db_session, "user-1", limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1


async def test_count_by_user(db_session):
    _make_user(db_session)
    _make_user(db_session, "user-2")
    await db_session.flush()

    await create(db_session, user_id="user-1", element_type="axel")
    await create(db_session, user_id="user-1", element_type="waltz_jump")
    await create(db_session, user_id="user-2", element_type="axel")

    count = await count_by_user(db_session, "user-1")
    assert count == 2


async def test_count_by_user_filter_element_type(db_session):
    _make_user(db_session)
    await db_session.flush()

    await create(db_session, user_id="user-1", element_type="axel")
    await create(db_session, user_id="user-1", element_type="waltz_jump")

    count = await count_by_user(db_session, "user-1", element_type="axel")
    assert count == 1


async def test_update_session(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    updated = await update(
        db_session,
        session,
        status="completed",
        overall_score=8.5,
    )
    assert updated.status == "completed"
    assert updated.overall_score == 8.5


async def test_update_session_with_metrics(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    metric = SessionMetric(
        session_id=session.id,
        metric_name="airtime",
        metric_value=0.65,
    )
    db_session.add(metric)
    await db_session.flush()

    updated = await update(db_session, session, status="completed")
    # metrics should be loaded via selectinload
    assert len(updated.metrics) == 1


async def test_update_session_ignores_none(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel", status="uploading")
    updated = await update(db_session, session, status=None, overall_score=7.0)
    # status should remain unchanged because None was passed
    assert updated.status == "uploading"
    assert updated.overall_score == 7.0


async def test_soft_delete(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    await soft_delete(db_session, session)

    fetched = await get_by_id(db_session, session.id)
    assert fetched.status == "deleted"


async def test_update_session_analysis(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")
    session_id = session.id

    pose_data = {"frames": [0, 1, 2], "fps": 30}
    frame_metrics = {"airtime": 0.65, "height": 0.42}
    phases = {"takeoff": 10, "peak": 20, "landing": 30}

    updated = await update_session_analysis(
        db_session,
        session_id=session_id,
        pose_data=pose_data,
        frame_metrics=frame_metrics,
        phases=phases,
    )
    assert updated.status == "completed"
    assert updated.pose_data == pose_data
    assert updated.frame_metrics == frame_metrics
    assert updated.phases == phases
    assert updated.processed_at is not None


async def test_update_session_analysis_none_values(db_session):
    _make_user(db_session)
    await db_session.flush()

    session = await create(db_session, user_id="user-1", element_type="axel")

    updated = await update_session_analysis(
        db_session,
        session_id=session.id,
        pose_data=None,
        frame_metrics=None,
        phases=None,
    )
    assert updated.status == "completed"
    assert updated.pose_data is None
    assert updated.frame_metrics is None
    assert updated.phases is None
