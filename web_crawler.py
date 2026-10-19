import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from urllib.parse import urlparse

# Mock HTML parser & network provider interface typically provided in the interview
class HtmlParser:
    def get_urls(self, url: str) -> list[str]:
        # Simulates fetching a page and extracting absolute URLs
        time.sleep(0.05)  # Simulate network latency
        if "seed" in url:
            return ["https://openai.com", "https://openai.com"]
        elif "about" in url:
            return ["https://openai.com", "https://openai.com"]
        return []

class MultithreadedWebCrawler:
    def __init__(self, parser: HtmlParser, max_workers: int = 5, rate_limit_delay: float = 0.1):
        self.parser = parser
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay

        # Concurrency primitives
        self.visited = set()
        self.visited_lock = threading.Lock()

        # Domain-based rate limiting tracking (Domain -> Last Access Timestamp)
        self.domain_locks = threading.Lock()
        self.domain_last_accessed = {}

    def _get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

    def _polite_delay(self, url: str):
        """Ensures threads respect domain rate-limits to avoid hitting servers too fast."""
        domain = self._get_domain(url)
        if not domain:
            return

        with self.domain_locks:
            last_access = self.domain_last_accessed.get(domain, 0)
            now = time.time()
            time_to_wait = (last_access + self.rate_limit_delay) - now

            if time_to_wait > 0:
                time.sleep(time_to_wait)
                now = time.time()

            self.domain_last_accessed[domain] = now

    def _process_url(self, url: str, depth: int, max_depth: int) -> list[tuple[str, int]]:
        """Worker task executing network fetch and link extraction safely."""
        if depth > max_depth:
            return []

        # Polite throttling per domain
        self._polite_delay(url)

        try:
            print(f"[Crawl] Fetching: {url} at depth {depth}")
            extracted_urls = self.parser.get_urls(url)
        except Exception as e:
            print(f"[Error] Failed to fetch {url}: {e}")
            return []

        next_tasks = []
        with self.visited_lock:
            for next_url in extracted_urls:
                if next_url not in self.visited:
                    self.visited.add(next_url)
                    next_tasks.append((next_url, depth + 1))

        return next_tasks

    def crawl(self, seed_url: str, max_depth: int):
        """Orchestrates the crawler lifecycle using an optimized worker pool loop."""
        print(f"Starting crawl from seed: {seed_url} (Max Depth: {max_depth})")

        with self.visited_lock:
            self.visited.add(seed_url)

        # Active tasks tracking (URL, depth)
        futures_map = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial seed
            initial_future = executor.submit(self._process_url, seed_url, 0, max_depth)
            futures_map[initial_future] = seed_url

            while futures_map:
                # Blocks until at least one thread completes a page pull
                done_futures = [f for f in as_completed(futures_map) if f.done()]

                for future in done_futures:
                    origin_url = futures_map.pop(future)
                    try:
                        next_tasks = future.result()
                        # Dynamic job submission: Spawn workers for newly discovered, unique URLs
                        for next_url, next_depth in next_tasks:
                            if next_depth <= max_depth:
                                f = executor.submit(self._process_url, next_url, next_depth, max_depth)
                                futures_map[f] = next_url
                    except Exception as exc:
                        print(f"[Error] Job generated an exception for {origin_url}: {exc}")

        print(f"\nCrawl complete! Total unique URLs discovered: {len(self.visited)}")
        return self.visited

# Execution Block
if __name__ == "__main__":
    parser = HtmlParser()
    crawler = MultithreadedWebCrawler(parser=parser, max_workers=3, rate_limit_delay=0.2)
    discovered = crawler.crawl("https://openai.com", max_depth=2)
