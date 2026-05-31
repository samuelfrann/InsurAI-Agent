import os
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import unquote

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

os.makedirs("naicom_docs", exist_ok=True)

# ── PDFs ─────────────────────────────────────────────────────────────
PDFS = [
    "https://naicom.gov.ng/wp-content/uploads/2025/08/NIIRA-2025.pdf",
    "https://nigeriainsurers.org/wp-content/uploads/2023/05/publications_Digest2016.pdf",
    "http://www.nigeriainsurers.org/publications/NIA-DIGEST-BOOK.pdf",
]

for pdf_url in PDFS:
    filename = unquote(pdf_url.split("/")[-1])
    path = os.path.join("naicom_docs", filename)
    if os.path.exists(path):
        print(f"  skip: {filename}")
        continue
    print(f"  downloading: {filename}")
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"    saved ({len(r.content) // 1024} KB)")
    except Exception as e:
        print(f"    FAILED: {e}")

# ── Web pages scraped as text ─────────────────────────────────────────
NIA_PAGES = [
    ("https://nigeriainsurers.org/faqs/",       "NIA_FAQs.txt"),
    ("https://nigeriainsurers.org/",             "NIA_About.txt"),
]

for url, filename in NIA_PAGES:
    path = os.path.join("naicom_docs", filename)
    if os.path.exists(path):
        print(f"  skip: {filename}")
        continue
    print(f"  scraping: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n\n{text}")
        print(f"    saved ({len(text) // 1024} KB text)")
    except Exception as e:
        print(f"    FAILED: {e}")

print(f"\nDone. Files in naicom_docs/: {len(os.listdir('naicom_docs'))}")