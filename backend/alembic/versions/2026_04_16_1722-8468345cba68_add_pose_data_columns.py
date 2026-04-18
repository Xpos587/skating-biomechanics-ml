"""add_pose_data_columns

Revision ID: 8468345cba68
Revises: aac592d509a9
Create Date: 2026-04-16 17:22:00.035500

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8468345cba68"
down_revision: str | Sequence[str] | None = "aac592d509a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new JSON columns for pose data and frame metrics
    op.add_column("sessions", sa.Column("pose_data", sa.JSON(), nullable=True))
    op.add_column("sessions", sa.Column("frame_metrics", sa.JSON(), nullable=True))
    # Note: Keeping old columns for backward compatibility
    # poses_url and csv_url will be deprecated in future migration


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sessions", "frame_metrics")
    op.drop_column("sessions", "pose_data")
