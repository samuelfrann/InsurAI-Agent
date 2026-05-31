import io
import pdfplumber
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import base64
from datetime import datetime

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool, Tool

# ── Shared in-process file store ─────────────────────────────────────────────
# The upload endpoint writes here; pdf_reader_tool reads from here.
# Works fine for single-worker. If you ever run gunicorn with multiple workers,
# swap this for a Redis-backed store.
file_store: dict = {}


@tool
def pdf_reader_tool(file_key: str) -> str:
    """Read and extract text and tables from an uploaded PDF document.
    Use this whenever a staff member uploads a PDF file and asks about its contents."""
    entry = file_store.get(file_key)
    if not entry:
        return "File not found or already processed. Ask the staff member to re-upload the document."

    try:
        with pdfplumber.open(io.BytesIO(entry["bytes"])) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                tables = page.extract_tables()
                section = f"── Page {i} ──\n{text}" if text else f"── Page {i} ── (no text)"
                for table in tables:
                    rows = [" | ".join(str(c or "").strip() for c in row) for row in table]
                    section += "\n\nTable:\n" + "\n".join(rows)
                pages.append(section)

        del file_store[file_key]
        result = f"Document: {entry['filename']}\n\n" + "\n\n".join(pages)
        return result[:8000]

    except Exception as e:
        return f"Failed to read PDF: {str(e)}"


def analyze_photo_exif(image_data: str, file_name: str = "") -> dict:
    """
    Extract and analyze EXIF metadata from a base64 image.
    Returns a structured fraud signal report.
    """
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))

        exif_data = img._getexif()

        if not exif_data:
            return {
                "has_metadata": False,
                "integrity": "unverified",
                "flag": "warning",
                "message": "No metadata found — photo may have been shared via WhatsApp or edited. Ask claimant to email the original photo directly.",
                "details": {}
            }

        # Parse EXIF tags
        parsed = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            parsed[tag] = value

        # Extract key fields
        details = {}

        # Timestamp
        date_taken = parsed.get("DateTimeOriginal") or parsed.get("DateTime")
        if date_taken:
            details["timestamp"] = date_taken
            details["timestamp_readable"] = date_taken

        # GPS
        gps_info = parsed.get("GPSInfo")
        if gps_info:
            gps_parsed = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            details["gps_present"] = True
            details["gps_raw"] = str(gps_parsed)
        else:
            details["gps_present"] = False

        # Device
        details["device"] = parsed.get("Model", "Unknown device")
        details["manufacturer"] = parsed.get("Make", "Unknown")

        # Software / editing
        software = parsed.get("Software", "")
        details["software"] = software
        edit_keywords = ["photoshop", "lightroom", "snapseed", "picsart", "canva", "gimp"]
        details["edit_detected"] = any(k in software.lower() for k in edit_keywords)

        # Build fraud signal
        flags = []
        if details.get("edit_detected"):
            flags.append("Photo shows signs of editing software")
        if not details.get("gps_present"):
            flags.append("No GPS location in photo")
        if not details.get("timestamp"):
            flags.append("No timestamp found")

        integrity = "verified" if not flags else "suspicious" if len(flags) >= 2 else "partial"

        return {
            "has_metadata": True,
            "integrity": integrity,
            "flag": "clear" if integrity == "verified" else "warning" if integrity == "partial" else "danger",
            "message": "Metadata intact." if not flags else " | ".join(flags),
            "details": details
        }

    except Exception as e:
        return {
            "has_metadata": False,
            "integrity": "error",
            "flag": "warning",
            "message": f"Could not read photo metadata: {str(e)}",
            "details": {}
        }


search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for information.",
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=100)
)