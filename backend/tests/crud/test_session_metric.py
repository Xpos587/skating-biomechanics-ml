"""Tests for SessionMetric CRUD operations."""

from __future__ import annotations

from app.crud.session_metric import bulk_create, get_current_best, get_current_best_batch
from app.models.session import Session, SessionMetric
from app.models.user import User


def _make_user(db, user_id: str = "user-1") -> User:
    user = User(id=user_id, email=f"{user_id}@test.com", hashed_password="hash")
    db.add(user)
    return user


async def test_get_current_best_returns_max(db_session):
    _make_user(db_session)
    await db_session.flush()

    # Create two sessions with different metric values
    s1 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    s2 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    db_session.add(s1)
    db_session.add(s2)
    await db_session.flush()

    db_session.add(SessionMetric(session_id=s1.id, metric_name="airtime", metric_value=0.5))
    db_session.add(SessionMetric(session_id=s2.id, metric_name="airtime", metric_value=0.8))
    await db_session.flush()

    best = await get_current_best(db_session, "user-1", "waltz_jump", "airtime")
    assert best == 0.8


async def test_get_current_best_no_data(db_session):
    _make_user(db_session)
    await db_session.flush()

    best = await get_current_best(db_session, "user-1", "waltz_jump", "airtime")
    assert best is None


async def test_get_current_best_ignores_deleted_sessions(db_session):
    _make_user(db_session)
    await db_session.flush()

    s1 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    s2 = Session(user_id="user-1", element_type="waltz_jump", status="deleted")
    db_session.add(s1)
    db_session.add(s2)
    await db_session.flush()

    db_session.add(SessionMetric(session_id=s1.id, metric_name="airtime", metric_value=0.5))
    db_session.add(SessionMetric(session_id=s2.id, metric_name="airtime", metric_value=0.9))
    await db_session.flush()

    best = await get_current_best(db_session, "user-1", "waltz_jump", "airtime")
    # Should only see the "done" session value
    assert best == 0.5


async def test_get_current_best_ignores_wrong_element_type(db_session):
    _make_user(db_session)
    await db_session.flush()

    s1 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    db_session.add(s1)
    await db_session.flush()

    db_session.add(SessionMetric(session_id=s1.id, metric_name="airtime", metric_value=0.5))
    await db_session.flush()

    best = await get_current_best(db_session, "user-1", "axel", "airtime")
    assert best is None


async def test_get_current_best_batch_real_db(db_session):
    _make_user(db_session)
    await db_session.flush()

    s1 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    s2 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    db_session.add(s1)
    db_session.add(s2)
    await db_session.flush()

    db_session.add(SessionMetric(session_id=s1.id, metric_name="airtime", metric_value=0.5))
    db_session.add(SessionMetric(session_id=s2.id, metric_name="airtime", metric_value=0.8))
    db_session.add(SessionMetric(session_id=s1.id, metric_name="max_height", metric_value=0.3))
    await db_session.flush()

    result = await get_current_best_batch(
        db_session, "user-1", "waltz_jump", ["airtime", "max_height"]
    )
    assert result == {"airtime": 0.8, "max_height": 0.3}


async def test_get_current_best_batch_empty_list(db_session):
    result = await get_current_best_batch(db_session, "user-1", "waltz_jump", [])
    assert result == {}


async def test_get_current_best_batch_no_data(db_session):
    _make_user(db_session)
    await db_session.flush()

    result = await get_current_best_batch(db_session, "user-1", "waltz_jump", ["airtime"])
    assert result == {}


async def test_bulk_create(db_session):
    _make_user(db_session)
    await db_session.flush()

    s1 = Session(user_id="user-1", element_type="waltz_jump", status="done")
    db_session.add(s1)
    await db_session.flush()

    await bulk_create(
        db_session,
        [
            {"session_id": s1.id, "metric_name": "airtime", "metric_value": 0.5},
            {"session_id": s1.id, "metric_name": "max_height", "metric_value": 0.3},
        ],
    )

    from sqlalchemy import select

    result = await db_session.execute(
        select(SessionMetric).where(SessionMetric.session_id == s1.id)
    )
    metrics = result.scalars().all()
    assert len(metrics) == 2
    names = {m.metric_name for m in metrics}
    assert names == {"airtime", "max_height"}
