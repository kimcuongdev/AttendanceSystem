-- Bảng lưu embedding khuôn mặt
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    base64_string TEXT NOT NULL,
    embedding BLOB NOT NULL,        -- embedding lưu dạng nhị phân
);
