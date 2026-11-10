from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Membuat tabel predictions beserta indeksnya."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("age", sa.Float(), nullable=False),
        sa.Column("anaemia", sa.SmallInteger(), nullable=False),
        sa.Column("creatinine_phosphokinase", sa.Float(), nullable=False),
        sa.Column("diabetes", sa.SmallInteger(), nullable=False),
        sa.Column("ejection_fraction", sa.Float(), nullable=False),
        sa.Column("high_blood_pressure", sa.SmallInteger(), nullable=False),
        sa.Column("platelets", sa.Float(), nullable=False),
        sa.Column("serum_creatinine", sa.Float(), nullable=False),
        sa.Column("serum_sodium", sa.Float(), nullable=False),
        sa.Column("sex", sa.SmallInteger(), nullable=False),
        sa.Column("smoking", sa.SmallInteger(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_category", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"], unique=False)
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"], unique=False)


def downgrade() -> None:
    """Menghapus tabel predictions jika rollback dilakukan."""
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")