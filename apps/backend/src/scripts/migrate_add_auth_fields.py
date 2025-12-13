"""
Migration script to add authentication fields to the database.
Run this script to add the new auth columns to existing databases.
"""
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sqlalchemy import text
from src.infrastructure.database import engine, Base
from src.infrastructure import models  # Import to register all models

def migrate():
    """Add authentication fields to users table and create refresh_tokens table"""
    print("Running authentication migration...")

    with engine.connect() as conn:
        # Check if columns already exist before adding
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'google_id'
        """))

        if result.fetchone() is None:
            print("Adding auth fields to users table...")

            # Add google_id column
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS google_id VARCHAR UNIQUE
            """))

            # Add phone_verified column
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS phone_verified BOOLEAN DEFAULT FALSE
            """))

            # Add auth_provider column
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS auth_provider VARCHAR DEFAULT 'local'
            """))

            # Add updated_at column
            conn.execute(text("""
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """))

            # Create index on google_id
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_users_google_id ON users (google_id)
            """))

            conn.commit()
            print("Auth fields added to users table.")
        else:
            print("Auth fields already exist in users table.")

        # Check if refresh_tokens table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'refresh_tokens'
            )
        """))

        if not result.fetchone()[0]:
            print("Creating refresh_tokens table...")

            conn.execute(text("""
                CREATE TABLE refresh_tokens (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    token_hash VARCHAR NOT NULL UNIQUE,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    revoked BOOLEAN DEFAULT FALSE
                )
            """))

            # Create indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_refresh_tokens_token_hash ON refresh_tokens (token_hash)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_refresh_tokens_user_id ON refresh_tokens (user_id)
            """))

            conn.commit()
            print("refresh_tokens table created.")
        else:
            print("refresh_tokens table already exists.")

    print("Migration completed successfully!")


if __name__ == "__main__":
    migrate()
