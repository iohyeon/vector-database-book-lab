"""5장 응용 실습: 실제 문장 임베딩을 PostgreSQL + pgvector에서 검색한다.

예제 자료는 이 저장소의 자체 작성 고객 지원 문서다. 작은 코퍼스이므로
벡터 전체를 비교하는 정확 검색을 쓰고, 카테고리 조건을 같은 SQL에 적용한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from fastembed import TextEmbedding


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "support_articles.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lab:lab_only_local_password@127.0.0.1:55443/vector_book_lab",
)


def vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def load_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(ROOT / ".cache" / "fastembed"),
        threads=2,
    )


def create_schema(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS article_chunks (
                article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                PRIMARY KEY (article_id, chunk_number)
            )
            """
        )


def build(connection: psycopg.Connection, model: TextEmbedding) -> tuple[int, int]:
    articles = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    chunks = [
        (article["id"], number, paragraph, f'{article["title"]}. {paragraph}')
        for article in articles
        for number, paragraph in enumerate(article["paragraphs"], start=1)
    ]
    embeddings = list(model.embed([chunk[3] for chunk in chunks]))
    if len(embeddings) != len(chunks):
        raise ValueError("The embedding model did not return one vector per chunk")
    if any(len(embedding) != MODEL_DIMENSION for embedding in embeddings):
        raise ValueError("The embedding model returned an unexpected dimension")

    with connection.cursor() as cursor:
        # 이 DB는 compose.yaml로 만든 전용 데모 DB다. 반복 실행 시 같은 결과를 만든다.
        cursor.execute("DELETE FROM article_chunks")
        cursor.execute("DELETE FROM articles")
        cursor.executemany(
            "INSERT INTO articles (id, title, category) VALUES (%s, %s, %s)",
            [(article["id"], article["title"], article["category"]) for article in articles],
        )
        cursor.executemany(
            """
            INSERT INTO article_chunks (article_id, chunk_number, content, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            [
                (article_id, number, paragraph, vector_literal(embedding))
                for (article_id, number, paragraph, _), embedding in zip(chunks, embeddings)
            ],
        )
    connection.commit()
    return len(articles), len(chunks)


def semantic_search(
    connection: psycopg.Connection,
    model: TextEmbedding,
    query: str,
    category: str | None = None,
    limit: int = 3,
) -> list[tuple]:
    query_vector = vector_literal(next(model.embed([query])))
    where = "WHERE a.category = %s" if category else ""
    parameters = [query_vector]
    if category:
        parameters.append(category)
    parameters.extend((query_vector, limit))
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT a.id, a.title, a.category, c.content,
                   1 - (c.embedding <=> %s::vector) AS cosine_similarity
            FROM article_chunks c
            JOIN articles a ON a.id = c.article_id
            {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            parameters,
        )
        return cursor.fetchall()


def keyword_search(connection: psycopg.Connection, query: str, limit: int = 3) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT a.id, a.title
            FROM article_chunks c
            JOIN articles a ON a.id = c.article_id
            WHERE to_tsvector('english', a.title || ' ' || c.content)
                  @@ plainto_tsquery('english', %s)
            LIMIT %s
            """,
            (query, limit),
        )
        return cursor.fetchall()


def display(query: str, keyword_matches: list[tuple], semantic_matches: list[tuple], category: str | None) -> None:
    print(f'Query: "{query}"')
    print(f"Category filter: {category or 'none'}")
    print(f"Keyword matches: {len(keyword_matches)}")
    for rank, (_, title, item_category, content, similarity) in enumerate(semantic_matches, start=1):
        print(f"  {rank}. {title} [{item_category}] cosine={similarity:.3f}")
        print(f"     {content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "search", "demo"))
    parser.add_argument("--query", default="I want my money back.")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    model = load_model()
    with psycopg.connect(DATABASE_URL) as connection:
        if args.command in ("build", "demo"):
            create_schema(connection)
            article_count, chunk_count = build(connection, model)
            print("PostgreSQL + pgvector | semantic support search")
            print(f"Model: {MODEL_NAME} (ONNX, {MODEL_DIMENSION} dimensions)")
            print(f"Indexed: {article_count} articles / {chunk_count} chunks")
            print("Index: exact scan (small corpus)")
            print("-" * 72)
        if args.command in ("search", "demo"):
            keywords = keyword_search(connection, args.query)
            semantic = semantic_search(connection, model, args.query, args.category)
            display(args.query, keywords, semantic, args.category)
        if args.command == "demo":
            print("-" * 72)
            billing_query = "Where can I get a receipt for my order?"
            keywords = keyword_search(connection, billing_query)
            semantic = semantic_search(connection, model, billing_query, category="billing")
            display(billing_query, keywords, semantic, category="billing")


if __name__ == "__main__":
    main()
