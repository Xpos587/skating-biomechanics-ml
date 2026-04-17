"""relationships_to_connections

Revision ID: 7e23d891c6d7
Revises: 1541cafaf37d
Create Date: 2026-04-17 19:08:25.185111

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "7e23d891c6d7"
down_revision: Union[str, Sequence[str], None] = "1541cafaf37d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop old partial unique index
    op.drop_index("uq_coach_skater_active", table_name="relationships")

    # 2. Rename table
    op.rename_table("relationships", "connections")

    # 3. Rename columns
    op.alter_column("connections", "coach_id", new_column_name="from_user_id")
    op.alter_column("connections", "skater_id", new_column_name="to_user_id")

    # 4. Add connection_type column (default 'coaching' for existing rows)
    op.add_column(
        "connections",
        sa.Column("connection_type", sa.String(length=20), nullable=False, server_default="coaching"),
    )


def downgrade() -> None:
    # 1. Remove connection_type column
    op.drop_column("connections", "connection_type")

    # 2. Rename columns back
    op.alter_column("connections", "from_user_id", new_column_name="coach_id")
    op.alter_column("connections", "to_user_id", new_column_name="skater_id")

    # 3. Rename table back
    op.rename_table("connections", "relationships")

    # 4. Recreate old partial unique index
    op.create_index(
        "uq_coach_skater_active",
        "relationships",
        ["coach_id", "skater_id"],
        unique=True,
        postgresql_where=sa.text("status != 'ended'"),
    )
