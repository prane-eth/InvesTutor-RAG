import time
from investutor.utils.retrieval_utils import search_documents, embeddings, rerank_results
from investutor.utils.ingestion import ingest_urls
from investutor.utils.web_search import search_web
from investutor.utils.ingestion import text_splitter


def check_ingestion_latency():
    urls = search_web("Investment")
    num_docs, time_taken = ingest_urls(urls)
    average_latency = time_taken / num_docs if num_docs > 0 else float('inf')
    print(f"Ingestion complete: inserted {num_docs} documents in {format_time(time_taken)}.")
    print(f"Average latency per document during ingestion: {format_time(average_latency)}")


sample_queries = [
    "What is investment?",  # 1
    "What is the role of a financial advisor in investments?",  # 2
    "What is the impact of inflation on investments?",  # 3
    "What is the significance of liquidity in investments?",  # 4
    "What are the risks associated with investments?",  # 5
    "What are the tax implications of different investment options?",  # 6
    "What are the latest trends in the investment market?",  # 7
    "What are the benefits of long-term investing?",  # 8
    "What are the key factors to consider before making an investment?",  # 9
    "What are the differences between stocks and bonds?",  # 10
    "What are the common mistakes to avoid in investing?",  # 12
    "What are mutual funds and how do they work?",  # 11
    "Explain different types of investments.",  # 13
    "Explain the concept of compound interest in investments.",  # 14
    "Explain socially responsible investing.",  # 15
    "Explain exchange-traded funds (ETFs).",  # 16
    "How does diversification help in investment?",  # 17
    "How to start investing with a small amount of money?",  # 18
    "How to evaluate the performance of an investment portfolio?",  # 19
    "How to balance risk and return in investments?",  # 20
    "How to use dollar-cost averaging in investments?",  # 21
    "How to create a diversified investment portfolio?",  # 22
    "How to invest in real estate?",  # 23
    "How to assess the risk tolerance for investments?",  # 24
    "How to stay updated with market trends for better investment decisions?",  # 25
]

def check_search_rerank_latency():
    search_count = 0
    start = time.process_time()
    for _ in range(2):
        for query in sample_queries:
            _ = search_documents(query)
            search_count += 1
    end = time.process_time()
    avg_latency = (end - start) / search_count

    print(f"Average latency per search+re-rank: {format_time(avg_latency)}")


def check_embedding_latency():
    # Measure latency for single-query embeddings and batch document embeddings

    # Measure embed_query latency (multiple runs)
    start = time.process_time()
    for q in sample_queries:
        _ = embeddings.embed_query(q)
    total_q_time = time.process_time() - start
    avg_query_time_per_item = total_q_time / len(sample_queries)
    print(f"embed_query: avg time per query: {format_time(avg_query_time_per_item)}")

    # Measure embed_documents latency for a batch
    batch_texts = [f"Document sample {i} about investing." for i in range(20)]
    start = time.process_time()
    _ = embeddings.embed_documents(batch_texts)
    batch_time = time.process_time() - start
    avg_batch_time_per_doc = batch_time / len(batch_texts)
    print(f"embed_documents: avg time per document: {format_time(avg_batch_time_per_doc)}")


def check_reranking_latency():
    # Measure latency for the reranking step using the existing search_documents
    if not sample_queries:
        print("Rerank: No rerank calls can made. No sample queries available.")
        return

    sample_query_results = []
    for q in sample_queries:
        # Get initial results (may be empty if vectorstore not configured)
        results = search_documents(q, k=10)  # These results are re-ranked, but do again.
        sample_query_results.append(results)

    start = time.process_time()
    for query, results in zip(sample_queries, sample_query_results):
        _ = rerank_results(query, results, top_k=3)
    total_rerank_time = time.process_time() - start
    total_rerank_calls = len(sample_queries)

    print(f"Rerank: avg time per call: {format_time(total_rerank_time/total_rerank_calls)}")
    # print(f"Rerank: Search: avg time per call: {format_time(total_search_time/total_rerank_calls)}")


def format_time(seconds: float) -> str:
    """Format seconds to a human-friendly string like '12 ms' or '1.234 s'."""
    if seconds is None:
        return "n/a"
    if seconds >= 1:
        return f"{seconds:.3f} s"
    return f"{seconds * 1000.0:.2f} ms"


def check_chunking_latency():
    """Measure latency of text chunking/splitting using the project's chunker.

    Runs timings on a single large synthetic document and on many
    smaller documents to estimate per-run and per-chunk timings.
    """
    paragraph = (
        "Investing involves allocating resources with the expectation of future returns. "
        "Diversification reduces risk by spreading investments across assets. "
    )

    # Large document test
    large_text = paragraph * 3000
    runs = 3
    total_time = 0.0
    total_chunks = 0
    for _ in range(runs):
        start = time.process_time()
        chunks = text_splitter.split_text(large_text)
        elapsed = time.process_time() - start
        total_time += elapsed
        total_chunks += len(chunks)

    avg_run_time = total_time / runs
    avg_chunks_per_run = total_chunks // runs if runs else 0
    avg_time_per_chunk = avg_run_time / avg_chunks_per_run if avg_chunks_per_run else float('inf')

    print(f"Chunking (large): avg run time: {format_time(avg_run_time)}")
    print(f"Chunking (large): avg chunks per run: {avg_chunks_per_run}")
    print(f"Chunking (large): avg time per chunk: {format_time(avg_time_per_chunk)}")

    # Many small documents test
    small_docs = [paragraph * 3 for _ in range(200)]
    start = time.process_time()
    all_chunks = []
    for doc in small_docs:
        c = text_splitter.split_text(doc)
        all_chunks.extend(c)
    elapsed = time.process_time() - start
    avg_per_doc = elapsed / len(small_docs) if small_docs else float('inf')
    avg_per_chunk = elapsed / len(all_chunks) if all_chunks else float('inf')

    print(f"Chunking (many small): total time {format_time(elapsed)} for {len(small_docs)} docs")
    print(f"Chunking (many small): avg time per doc: {format_time(avg_per_doc)}")
    print(f"Chunking (many small): avg time per chunk: {format_time(avg_per_chunk)}")


def main():
    print()
    check_embedding_latency()
    print()
    check_reranking_latency()
    print()
    check_ingestion_latency()
    print()
    check_search_rerank_latency()
    print()
    check_chunking_latency()
    print()


if __name__ == "__main__":
    main()
