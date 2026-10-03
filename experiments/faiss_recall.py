"""3장 실습: 정확 검색과 근사 검색의 속도·재현율을 함께 측정한다.

책의 3장 FAISS 벤치마크에 recall@k를 더한 재현용 예제다.
합성 난수 벡터를 사용하므로 결과를 의미 검색의 품질로 해석하면 안 된다.
"""

from __future__ import annotations

import platform
import statistics
import time

import faiss
import numpy as np


SEED = 20261003
DIMENSION = 64
VECTOR_COUNT = 10_000
QUERY_COUNT = 100
TOP_K = 5
CLUSTER_COUNT = 100
REPEATS = 3


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(set(row) & set(target)) for row, target in zip(found, truth))
    return hits / truth.size


def measure(index: faiss.Index, queries: np.ndarray) -> tuple[np.ndarray, float]:
    # 전체 쿼리를 묶어서 처리하는 배치 시간보다 사용자 요청 1건의 지연을 본다.
    index.search(queries[:1], TOP_K)
    elapsed = []
    found = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        rows = [index.search(query.reshape(1, -1), TOP_K)[1][0] for query in queries]
        elapsed.append((time.perf_counter() - start) * 1_000 / len(queries))
        found = np.asarray(rows)
    assert found is not None
    return found, statistics.median(elapsed)


def main() -> None:
    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(SEED)
    vectors = rng.random((VECTOR_COUNT, DIMENSION), dtype=np.float32)
    queries = rng.random((QUERY_COUNT, DIMENSION), dtype=np.float32)

    flat = faiss.IndexFlatL2(DIMENSION)
    flat.add(vectors)
    _, truth = flat.search(queries, TOP_K)

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIMENSION), DIMENSION, CLUSTER_COUNT)
    ivf.train(vectors)
    ivf.add(vectors)

    hnsw = faiss.IndexHNSWFlat(DIMENSION, 16)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(vectors)

    print("FAISS chapter 3 | recall vs. latency")
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | FAISS {faiss.__version__}")
    print(f"seed={SEED}, vectors={VECTOR_COUNT}, dimensions={DIMENSION}, queries={QUERY_COUNT}, k={TOP_K}")
    print("data=uniform random float32; latency=median of 3 runs, single-query ms")
    print("-" * 64)
    print(f"{'index / setting':<25} {'recall@5':>10} {'ms/query':>12}")

    found, latency = measure(flat, queries)
    assert recall_at_k(found, truth) == 1.0
    print(f"{'Flat (exact)':<25} {recall_at_k(found, truth):>10.3f} {latency:>12.4f}")

    for nprobe in (1, 4, 16, 100):
        ivf.nprobe = nprobe
        found, latency = measure(ivf, queries)
        if nprobe == CLUSTER_COUNT:
            assert recall_at_k(found, truth) == 1.0
        print(f"{f'IVF nprobe={nprobe}':<25} {recall_at_k(found, truth):>10.3f} {latency:>12.4f}")

    for ef_search in (8, 32, 128):
        hnsw.hnsw.efSearch = ef_search
        found, latency = measure(hnsw, queries)
        print(f"{f'HNSW efSearch={ef_search}':<25} {recall_at_k(found, truth):>10.3f} {latency:>12.4f}")


if __name__ == "__main__":
    main()
