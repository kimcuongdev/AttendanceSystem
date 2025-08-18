import sqlite3
from pathlib import Path
from config import DB_PATH, BASE_DIR, SCHEMA_PATH


class DatabaseConnection:
    """
    Quản lý kết nối SQLite bằng context manager.
    Dùng: with DatabaseConnection() as conn: ...
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()


def init_db():
    """Khởi tạo schema từ file schema.sql"""
    schema_file = SCHEMA_PATH
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = f.read()

    with DatabaseConnection() as conn:
        conn.executescript(schema)
