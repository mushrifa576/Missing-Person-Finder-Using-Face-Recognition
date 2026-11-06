
import os
import psycopg2
from config import DB_CONFIG, DATABASE_URL

def get_connection():
    if DATABASE_URL:
        return psycopg2.connect(
            DATABASE_URL,
            sslmode="require",
            connect_timeout=10
        )

    return psycopg2.connect(**DB_CONFIG)
