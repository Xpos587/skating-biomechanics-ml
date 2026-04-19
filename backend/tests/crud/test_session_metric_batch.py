"""Tests for batch get_current_best query."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def db_session():
    session = AsyncMock()
    return session


async def test_get_current_best_batch_returns_dict():
    from app.crud.session_metric import get_current_best_batch

    mock_row1 = MagicMock()
    mock_row1.metric_name = "airtime"
    mock_row1.best_value = 0.65
    mock_row2 = MagicMock()
    mock_row2.metric_name = "max_height"
    mock_row2.best_value = 0.42

    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row1, mock_row2]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await get_current_best_batch(
        db,
        user_id="user-1",
        element_type="waltz_jump",
        metric_names=["airtime", "max_height"],
    )
    assert result == {"airtime": 0.65, "max_height": 0.42}


async def test_get_current_best_batch_returns_empty_for_no_metrics():
    from app.crud.session_metric import get_current_best_batch

    mock_result = MagicMock()
    mock_result.all.return_value = []

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await get_current_best_batch(
        db,
        user_id="user-1",
        element_type="waltz_jump",
        metric_names=["airtime"],
    )
    assert result == {}
