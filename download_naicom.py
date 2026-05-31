# scripts/download_naicom.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Every page to scrape
SCRAPE_PAGES = [
    "https://naicom.gov.ng/regulatory-instruments/",
    "https://naicom.gov.ng/guidelines",
    "https://naicom.gov.ng/naicom-circulars",
    "https://naicom.gov.ng/law",
]

# All confirmed PDFs hardcoded as fallback — scraped from site directly
KNOWN_PDFS = [
    # ── Laws ──────────────────────────────────────────────────────────
    "https://naicom.gov.ng/wp-content/uploads/2026/04/NAICOM-Act-1997.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/10/NIGERIA-INDUSTRY-INSURANCE_052600.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/05/NAICOMs-Recapitalisation-Circular-NIIRA-2025.pdf",

    # ── Guidelines ────────────────────────────────────────────────────
    "https://naicom.gov.ng/wp-content/uploads/2026/05/Guidelines-for-the-Operation-of-Foreign-or-International-Health-Insurance-Reinsurance-Providers-FINAL-1.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/04/Guidelines-on-Insurance-Policyholders-Protection-Fund.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/01/Guidelines-on-Licensing-and-Renewal-for-Insurance-Institutions-in-Nigeria-FINAL.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/08/Approved-Bancassurance-Guidelines-2017-1.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/10/Approved-Regulatory-Sandbox-May-2023.docx.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/08/National-Insurance-Commission-Corporate-Governance-Guidelines-For-Insurance-And-Reinsurance-Companies-In-Nigeria.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/08/JANUARY-2024-REVISED-MARKET-CONDUCT-AND-BUSINESS-PRACTICE-GUIDELINES-FOR-INSURANCE-REINSURANCE-COMPANIES-JAN.-2024.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/08/4_REVISED-PRUDENTIAL-GUIDELINES-FOR-INS-INSTIT-IN-NIG-OCT-2022.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/07/Guidelines-for-Insurtech-Operations-in-Nigeria.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2025/05/Guideline-for-Insurance-of-Government-Assets-2.pdf",
    # Guidelines on storage.naicom.website (different domain — old script missed these)
    "https://storage.naicom.website/naicom/files/no%20premiun%20no%20cover%20circular.pdf",
    "https://storage.naicom.website/naicom/files/GUIDELINES%20ON%20WEBB%20AGGRE%202022.pdf",
    "https://storage.naicom.website/naicom/files/MicroInsurance%20Guidelines%202018.pdf",
    "https://storage.naicom.website/naicom/files/Prudential%20Guidelines%20For%20tnsurers%20and%20Reinsurers%20In%20Nigeria.pdf",
    "https://storage.naicom.website/naicom/files/Takaful%20Guidelines%20Approved.pdf",

    # ── Circulars ─────────────────────────────────────────────────────
    "https://naicom.gov.ng/wp-content/uploads/2026/05/Prohibition-of-Coinsurance-Arrangements-Between-Takaful-Companies-and-Conventional-Insurance-Companies-1.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/05/cirs-2022-to-2025.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/05/CIRCULAR-ON-ADVERT-JUNE-2025.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/04/Circular-on-Underwriting-of-Annuity-Biz-%E2%80%93-2025Final.pdf",
    "https://naicom.gov.ng/wp-content/uploads/2026/05/NAICOM-SLA.pdf",
]

os.makedirs("naicom_docs", exist_ok=True)

# Collect all PDF URLs — start with known list
all_pdfs = set(KNOWN_PDFS)

# Scrape pages dynamically to catch anything new
ALLOWED_DOMAINS = ("naicom.gov.ng", "storage.naicom.website")
for url in SCRAPE_PAGES:
    print(f"Scanning {url} ...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf") and any(d in href for d in ALLOWED_DOMAINS):
                all_pdfs.add(href)
    except Exception as e:
        print(f"  scan failed: {e}")

print(f"\nFound {len(all_pdfs)} PDFs total. Downloading...\n")

success, skipped, failed = 0, 0, 0

for pdf_url in sorted(all_pdfs):
    filename = unquote(pdf_url.split("/")[-1])
    path = os.path.join("naicom_docs", filename)

    if os.path.exists(path):
        print(f"  skip: {filename}")
        skipped += 1
        continue

    print(f"  downloading: {filename}")
    try:
        verify = False if "storage.naicom.website" in pdf_url else True
        r = requests.get(pdf_url, headers=HEADERS, timeout=30, verify=verify)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"    saved ({len(r.content) // 1024} KB)")
        success += 1
    except Exception as e:
        print(f"    FAILED: {e}")
        failed += 1

print(f"\nDone. {success} downloaded, {skipped} skipped, {failed} failed.")
print(f"Files in naicom_docs/: {len(os.listdir('naicom_docs'))}")