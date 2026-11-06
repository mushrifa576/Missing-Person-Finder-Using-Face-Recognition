import os
import psycopg2
from config import DATABASE_URL, DB_CONFIG


def get_connection():

    if DATABASE_URL:
        return psycopg2.connect(
            DATABASE_URL,
            sslmode=os.getenv("DB_SSLMODE", "require"),
            connect_timeout=10
        )

    return psycopg2.connect(
        **DB_CONFIG,
        connect_timeout=10
    )