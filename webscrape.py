import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time

# Configuration
DOMAINS = [
    "https://www.jewelchangiairport.com/",
    "https://www.changiairport.com/",
]
OUTPUT_FILE = "scraped_data.jsonl"
MAX_PAGES_PER_DOMAIN = 50  # limit pages per domain
CRAWL_DELAY = 1  # seconds between requests


def is_internal_link(link, base_netloc):
    parsed = urlparse(link)
    # Empty netloc means relative URL, treat as internal
    if not parsed.netloc:
        return True
    return parsed.netloc == base_netloc


def get_links(soup, base_url):
    links = set()
    base_netloc = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]  # strip fragments
        full_url = urljoin(base_url, href)
        if is_internal_link(full_url, base_netloc):
            links.add(full_url)
    return links


def extract_text(soup):
    # Extract text from title, headings, paragraphs
    texts = []
    if soup.title:
        texts.append(soup.title.get_text(strip=True))
    for tag in ["h1", "h2", "h3", "p"]:
        for el in soup.find_all(tag):
            txt = el.get_text(strip=True)
            if txt:
                texts.append(txt)
    return "\n".join(texts)


def scrape_site(base_url):
    visited = set()
    to_visit = {base_url}
    domain_data = []
    base_netloc = urlparse(base_url).netloc

    while to_visit and len(visited) < MAX_PAGES_PER_DOMAIN:
        url = to_visit.pop()
        if url in visited:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        text = extract_text(soup)
        domain_data.append({"url": url, "text": text})
        visited.add(url)
        print(f"Scraped ({len(visited)}) {url}")

        # Crawl new links
        for link in get_links(soup, base_url):
            if link not in visited and link not in to_visit:
                to_visit.add(link)

        time.sleep(CRAWL_DELAY)

    return domain_data


def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for domain in DOMAINS:
            print(f"Starting scrape for {domain}")
            pages = scrape_site(domain)
            for page in pages:
                fout.write(json.dumps(page, ensure_ascii=False) + "\n")
    print(f"Scraping completed. Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
