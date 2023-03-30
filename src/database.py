import sqlite3
import uuid

import numpy as np

from embedding import Embedding


class EmbeddingDatabase:
    def __init__(self, db_name='embeddings.sqlite'):
        self.db_name = db_name
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guid TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')
            conn.commit()

    def insert_embedding(self, embedding: Embedding):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            embedding_arr = np.array(embedding.embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (guid, text, embedding) VALUES (?, ?, ?)",
                (str(embedding.guid), embedding.text, embedding_arr)
            )
            conn.commit()
            return embedding.guid

    def get_embeddings(self) -> list[Embedding]:
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT guid, text, embedding FROM embeddings")
            result = cursor.fetchall()

        embeddings = []
        for row in result:
            guid, text, emb_blob = row
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            embeddings.append(Embedding(text, guid, embedding=embedding))

        return embeddings
