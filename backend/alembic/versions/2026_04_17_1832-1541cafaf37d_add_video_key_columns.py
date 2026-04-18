"""add_video_key_columns

Revision ID: 1541cafaf37d
Revises: 8468345cba68
Create Date: 2026-04-17 18:32:48.396480

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1541cafaf37d"
down_revision: str | Sequence | None = "8468345cba68"
branch_labels: str | Sequence | None = None
depends_on: str | Sequence | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("sessions", sa.Column("video_key", sa.String(500), nullable=True))
    op.add_column("sessions", sa.Column("processed_video_key", sa.String(500), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sessions", "processed_video_key")
    op.drop_column("sessions", "video_key")
