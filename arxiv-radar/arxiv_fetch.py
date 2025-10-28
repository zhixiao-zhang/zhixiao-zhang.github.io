# arxiv_fetch.py
#
# Fetch papers from arXiv using its public API.
# This module only fetches and normalizes results.
#
# Usage:
#   papers = query_arxiv(["cs.CR","cs.PL"], ["memory safety","pointer analysis"])

from typing import List, Dict
import requests
import feedparser

def query_arxiv(categories: List[str],
                max_results: int = 100) -> List[Dict]:
    """
    Query arXiv for recent papers in the given categories,
    WITHOUT applying any keyword filter yet.

    We'll later let AI decide relevance instead of doing keyword match.
    """

    # Category clause: (cat:cs.CR OR cat:cs.PL ...)
    cat_query = " OR ".join([f"cat:{c}" for c in categories])

    # NOTE: no kw_query here. We only restrict by category.
    search_query = f"({cat_query})"

    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={search_query}"
        f"&sortBy=lastUpdatedDate"
        f"&sortOrder=descending"
        f"&max_results={max_results}"
    )

    resp = requests.get(
        url,
        timeout=50,
        headers={
            "User-Agent": "arxiv-radar-script/0.2 (+https://github.com/yourname)"
        }
    )
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)

    papers: List[Dict] = []
    for entry in feed.entries:
        pdf_links = [
            l.href for l in entry.links
            if getattr(l, "type", "") == "application/pdf"
        ]
        pdf_url = pdf_links[0] if pdf_links else None

        arxiv_id = entry.id.split("/")[-1]

        papers.append({
            "id": entry.id,
            "arxiv_id": arxiv_id,
            "title": entry.title.strip(),
            "authors": [a.name for a in entry.authors],
            "summary": entry.summary.strip(),
            "updated": entry.updated,
            "published": entry.published,
            "pdf_url": pdf_url,
            "primary_category": entry.arxiv_primary_category["term"]
                                if hasattr(entry, "arxiv_primary_category") else None,
            "categories": [t["term"] for t in entry.tags] if hasattr(entry, "tags") else [],
        })

    return papers

