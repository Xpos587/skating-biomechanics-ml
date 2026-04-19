"""add_music_fingerprint

Revision ID: 7c318ee90e4a
Revises: 7e23d891c6d7
Create Date: 2026-04-19 14:48:22.268569

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c318ee90e4a"
down_revision: str | Sequence[str] | None = "7e23d891c6d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create enum types before using them in ALTER COLUMN
    connectiontype = sa.Enum("COACHING", "CHOREOGRAPHY", name="connectiontype", create_type=False)
    connectiontype.create(op.get_bind(), checkfirst=True)

    connectionstatus = sa.Enum(
        "INVITED", "ACTIVE", "ENDED", name="connectionstatus", create_type=False
    )
    connectionstatus.create(op.get_bind(), checkfirst=True)

    # Drop defaults first (VARCHAR default can't auto-cast to enum)
    op.execute("ALTER TABLE connections ALTER COLUMN connection_type DROP DEFAULT")
    op.execute("ALTER TABLE connections ALTER COLUMN status DROP DEFAULT")

    # VARCHAR → enum with explicit USING clause
    op.execute(
        "ALTER TABLE connections ALTER COLUMN connection_type TYPE connectiontype USING connection_type::connectiontype"
    )
    op.execute(
        "ALTER TABLE connections ALTER COLUMN status TYPE connectionstatus USING status::connectionstatus"
    )

    # Re-set defaults as enum values
    op.execute(
        "ALTER TABLE connections ALTER COLUMN connection_type SET DEFAULT 'COACHING'::connectiontype"
    )
    op.execute(
        "ALTER TABLE connections ALTER COLUMN status SET DEFAULT 'INVITED'::connectionstatus"
    )

    op.add_column("music_analyses", sa.Column("fingerprint", sa.String(length=32), nullable=True))
    op.create_index(
        op.f("ix_music_analyses_fingerprint"), "music_analyses", ["fingerprint"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_music_analyses_fingerprint"), table_name="music_analyses")
    op.drop_column("music_analyses", "fingerprint")
    op.alter_column(
        "connections",
        "status",
        existing_type=sa.Enum("INVITED", "ACTIVE", "ENDED", name="connectionstatus"),
        type_=sa.VARCHAR(length=20),
        existing_nullable=False,
    )
    op.alter_column(
        "connections",
        "connection_type",
        existing_type=sa.Enum("COACHING", "CHOREOGRAPHY", name="connectiontype"),
        type_=sa.VARCHAR(length=20),
        existing_nullable=False,
        existing_server_default=sa.text("'coaching'::character varying"),
    )
    sa.Enum(name="connectionstatus").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="connectiontype").drop(op.get_bind(), checkfirst=True)
