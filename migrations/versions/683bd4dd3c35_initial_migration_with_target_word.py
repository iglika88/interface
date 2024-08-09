"""Initial migration with target_word

Revision ID: 683bd4dd3c35
Revises:
Create Date: 2024-07-23 10:42:28.311340

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '683bd4dd3c35'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=20), nullable=False),
    sa.Column('email', sa.String(length=120), nullable=False),
    sa.Column('password_hash', sa.String(length=128), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_table('vocabulary_entries',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('item', sa.String(length=255), nullable=False),
    sa.Column('pos', sa.String(length=50), nullable=True),
    sa.Column('translation', sa.String(length=255), nullable=True),
    sa.Column('lesson_title', sa.String(length=255), nullable=True),
    sa.Column('reading_or_listening', sa.String(length=50), nullable=True),
    sa.Column('course_code', sa.String(length=50), nullable=False),
    sa.Column('cefr_level', sa.String(length=10), nullable=False),
    sa.Column('domain', sa.String(length=255), nullable=False),
    sa.Column('user', sa.String(length=50), nullable=False),
    sa.Column('date_loaded', sa.DateTime(), nullable=False),
    sa.Column('number_of_contexts', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('context_entry',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('item_id', sa.Integer(), nullable=False),
    sa.Column('context', sa.Text(), nullable=False),
    sa.Column('target_word', sa.String(length=128), nullable=False),
    sa.Column('user', sa.String(length=50), nullable=False),
    sa.Column('date_added', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['item_id'], ['vocabulary_entries.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('exercises',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('answer', sa.String(length=128), nullable=False),
    sa.Column('exercise_type', sa.String(length=50), nullable=False),
    sa.Column('date_created', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('exercises')
    op.drop_table('context_entry')
    op.drop_table('vocabulary_entries')
    op.drop_table('users')
    # ### end Alembic commands ###

