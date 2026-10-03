"""4장 실습: 원본 행 ID가 바뀌면 별도 인덱스와 연결이 끊기는 상황을 재현한다.

벡터 확장 설치 없이 SQLite 자체의 INSERT OR REPLACE / UPSERT 동작만 확인한다.
"""

import sqlite3


def setup() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            post_id TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL
        );
        CREATE TABLE vector_index (
            source_rowid INTEGER PRIMARY KEY,
            vector_label TEXT NOT NULL
        );
        INSERT INTO posts (post_id, content) VALUES ('p-1', 'old text');
        INSERT INTO vector_index (source_rowid, vector_label)
            SELECT id, 'old embedding' FROM posts WHERE post_id = 'p-1';
        """
    )
    return connection


def state(connection: sqlite3.Connection) -> tuple[int, int, bool]:
    post_rowid = connection.execute("SELECT id FROM posts WHERE post_id = 'p-1'").fetchone()[0]
    indexed_rowid = connection.execute("SELECT source_rowid FROM vector_index").fetchone()[0]
    return post_rowid, indexed_rowid, post_rowid == indexed_rowid


def main() -> None:
    replace_connection = setup()
    before = state(replace_connection)
    replace_connection.execute(
        "INSERT OR REPLACE INTO posts (post_id, content) VALUES (?, ?)",
        ("p-1", "new text"),
    )
    after_replace = state(replace_connection)

    upsert_connection = setup()
    upsert_connection.execute(
        """
        INSERT INTO posts (post_id, content) VALUES (?, ?)
        ON CONFLICT(post_id) DO UPDATE SET content = excluded.content
        """,
        ("p-1", "new text"),
    )
    after_upsert = state(upsert_connection)

    print("SQLite chapter 4 | rowid mapping")
    print(f"SQLite {sqlite3.sqlite_version} | in-memory database")
    print("vector_index is a minimal stand-in for posts_vss; no vector extension is loaded")
    print("-" * 67)
    print(f"{'operation':<20} {'posts.id':>10} {'index ID':>10} {'linked':>10}")
    for label, result in (
        ("before", before),
        ("INSERT OR REPLACE", after_replace),
        ("UPSERT", after_upsert),
    ):
        print(f"{label:<20} {result[0]:>10} {result[1]:>10} {str(result[2]):>10}")

    assert before[2]
    assert not after_replace[2]
    assert after_upsert[2]


if __name__ == "__main__":
    main()
