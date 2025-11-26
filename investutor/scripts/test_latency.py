import time
from investutor.utils.retrieval_utils import search_documents
from investutor.utils.ingestion import ingest_urls
from investutor.utils.web_search import search_web


def main():
	urls = search_web("Investment")
	num_docs, time_taken = ingest_urls(urls)
	print(f"Ingestion complete: inserted {num_docs} documents in {time_taken:.2f} seconds.")
	average_latency = time_taken / num_docs if num_docs > 0 else float('inf')
	print(f"Average latency per document during ingestion: {average_latency:.4f} seconds")

	results = []
	search_times = 10
	start = time.time()
	for _ in range(search_times):
		results = search_documents("What is investment?")
	end = time.time()

	print("Results:", len(results), f"documents for {search_times} searches.")
	print("Average latency per search:", (end - start) / search_times, "seconds")

if __name__ == "__main__":
	main()
