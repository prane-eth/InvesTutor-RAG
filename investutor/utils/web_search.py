
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Keys of all members of the team.
if not GOOGLE_API_KEY:
	raise ValueError("Google API key is not set in the environment.")

GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not GOOGLE_CSE_ID:
	raise ValueError("Google Programmable Search Engine ID (GOOGLE_CSE_ID) is not set in the environment.")

# max_fetch_results = os.getenv("SCRAPER_MAX_FETCH_RESULTS", "5")
# max_fetch_results = int(max_fetch_results)

service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)


def search_web(query: str) -> list[str]:
	try:
		# Perform a Google search using the provided query and return a list of URLs.
		res_data = service.cse().list(q=query, cx=GOOGLE_CSE_ID).execute()
		total_results = int(res_data.get("searchInformation", {}).get("totalResults", 0))

		# return res_data if total_results else None
		if not total_results:
			print("Google Search: Fetched results successfully: 0 results")
			return []
		items = res_data.get("items", [])
		if not items:
			print("Google Search: Fetched results successfully: 0 items")
			return []

		# items = items[:max_fetch_results]
		urls = [item.get("link") for item in items if item.get("link")]
		print("Google Search: Fetched results successfully:", len(urls), "URLs")
		return urls or []

	except Exception as e:
		if "429" in str(e):
			raise e
			# change_api_key()
			# return search_web(query)
		else:
			print(f"Google Search Error: {e}".split('key=')[0])
			raise e


if __name__ == "__main__":
	# Test the code
	try:
		query = "Python programming language"
		results = search_web(query)
		print("Search results for query:", query)
		for url in results:
			print(url)
	except Exception as e:
		print(f"An error occurred during Google search: {e}")
		raise e
