"""Discovery: find relevant papers NOT yet in the user's library.

While writing, Scholia can surface candidate papers from public scholarly search
APIs (Semantic Scholar Academic Graph + PubMed E-utilities) that are relevant to
a passage and are **not already in the user's Zotero library**, so they can be
validated (via the user's own external triple-validating ingester) and added.

INTEGRITY BOUNDARY
------------------
Discovery only *finds and suggests* papers. It never generates prose, never
writes a citation into the draft, and never adds anything to Zotero on its own —
the add path shells out to the user's own external triple-validating ingester
(configured via ``--ingest-cmd`` / ``SCHOLIA_INGEST_CMD``; Scholia ships none).

PRIVACY
-------
The user's draft never leaves the machine. ``build_query`` extracts a short
keyword string locally (simple stopword-filtered key-term extraction); only that
query string is sent to the search APIs. No cloud LLM is involved at any point.

NETWORK
-------
The real sources use the standard library (``urllib``) only — no new dependency.
Network access is lazy and guarded: a clear error is raised when offline so the
CLI can degrade gracefully. ``FakeDiscoverySource`` is fully deterministic and
offline for unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from scholia.models import Paper

# Contact address for API etiquette (NCBI E-utilities asks for one, Semantic
# Scholar's User-Agent likewise). This is NOT a personal address: it defaults to a
# generic project value and is overridable via the SCHOLIA_CONTACT_EMAIL env var so
# a user can advertise their own contact without editing source. Never ship the
# maintainer's personal email.
_DEFAULT_CONTACT_EMAIL = "scholia@users.noreply.github.com"


def _contact_email() -> str:
    """Resolve the contact email: SCHOLIA_CONTACT_EMAIL env, else generic default."""
    env = os.environ.get("SCHOLIA_CONTACT_EMAIL")
    return env.strip() if env and env.strip() else _DEFAULT_CONTACT_EMAIL

# --- Candidate record -------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A discovered paper that may or may not already be in the library.

    ``score`` is the source's own relevance score (higher = more relevant);
    ``source`` is the originating backend (e.g. ``"semanticscholar"``,
    ``"pubmed"``). ``abstract_snippet`` is a short excerpt for display only.
    """

    title: str
    authors: list[str]
    year: str
    doi: str
    abstract_snippet: str
    source: str
    score: float


# --- Source protocol --------------------------------------------------------


@runtime_checkable
class DiscoverySource(Protocol):
    """A backend that searches a scholarly corpus for candidate papers.

    The single required method is ``search(query, limit) -> list[Candidate]``.
    Implementations should be lazy about network access and raise a clear error
    (``DiscoveryUnavailable``) when they cannot reach their service.
    """

    def search(self, query: str, limit: int) -> list[Candidate]:
        ...


class DiscoveryUnavailable(RuntimeError):
    """Raised when a real source cannot reach its API (offline / rate-limited)."""


# --- Query construction (privacy-aware) -------------------------------------

# Minimal English stopword list — enough to strip filler so only content words
# (gene names, methods, concepts) reach the search API. Deliberately small and
# local; no external NLP dependency.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have here however
    if in into is it its many more most much of on or our that the their then
    there these this thus to very was we were what when which while with within
    study found shows show during clearly important field results result also
    using used use both than then them they not no can may will
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def build_query(passage: str, max_terms: int = 8) -> str:
    """Extract a short keyword query from a draft passage (privacy-aware).

    Only this short string is ever sent to a search API — never the raw draft.
    Key-term extraction is intentionally simple and fully local: tokenize, drop
    stopwords and very short tokens, keep the first ``max_terms`` distinct
    content words **in their original order** (deterministic, order-preserving).

    Returns ``""`` for an empty/whitespace passage.
    """
    if not passage or not passage.strip():
        return ""
    terms: list[str] = []
    seen: set[str] = set()
    for m in _WORD_RE.finditer(passage):
        tok = m.group(0)
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if len(low) < 3 and not low.isdigit():
            continue
        if low in seen:
            continue
        seen.add(low)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


# --- Dedup against the library ----------------------------------------------


def _norm_doi(doi: str) -> str:
    """Normalize a DOI for comparison: strip a doi.org prefix, lowercase, trim."""
    d = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _norm_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, collapse non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedupe_against_library(
    candidates: list[Candidate], library: list[Paper]
) -> list[Candidate]:
    """Return candidates NOT already in the library, also de-duped among themselves.

    A candidate is dropped if its (normalized) DOI matches a library paper's DOI,
    or — when it has no DOI — its (normalized) title matches a library title.
    Candidates are also de-duped against each other (same DOI, else same title),
    so the same paper returned by two sources collapses to its first occurrence.
    Order is preserved (first occurrence wins).
    """
    lib_dois = {_norm_doi(p.doi) for p in library if _norm_doi(p.doi)}
    lib_titles = {_norm_title(p.title) for p in library if _norm_title(p.title)}

    out: list[Candidate] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    for c in candidates:
        d = _norm_doi(c.doi)
        t = _norm_title(c.title)
        # Already in the library?
        if d and d in lib_dois:
            continue
        if not d and t and t in lib_titles:
            continue
        # Already seen among the candidates themselves?
        if d:
            if d in seen_dois:
                continue
            seen_dois.add(d)
        elif t:
            if t in seen_titles:
                continue
            seen_titles.add(t)
        out.append(c)
    return out


# --- Orchestration ----------------------------------------------------------


def discover(
    passage: str,
    sources: list[DiscoverySource],
    library: list[Paper],
    limit: int = 8,
) -> list[Candidate]:
    """Find NEW candidate papers relevant to ``passage``, ranked by score desc.

    Builds a privacy-aware keyword query from ``passage``, queries every source,
    merges and de-dupes the results, drops anything already in ``library``, sorts
    by descending source score, and truncates to ``limit``.

    A source that raises ``DiscoveryUnavailable`` is skipped (the others still
    contribute); if every source fails, the exception propagates so the caller
    can report being offline.
    """
    query = build_query(passage)
    if not query:
        return []

    merged: list[Candidate] = []
    failures = 0
    for src in sources:
        try:
            merged.extend(src.search(query, limit=max(limit, 1)))
        except DiscoveryUnavailable:
            failures += 1
            continue
    if sources and failures == len(sources):
        raise DiscoveryUnavailable(
            "All discovery sources were unreachable (offline or rate-limited)."
        )

    new = dedupe_against_library(merged, library)
    new.sort(key=lambda c: c.score, reverse=True)
    return new[:limit]


# --- FakeDiscoverySource (deterministic, offline; test-only) ----------------


class FakeDiscoverySource:
    """Deterministic, offline discovery source for unit tests.

    Derives a stable set of candidates from a SHA-256 hash of the query, so the
    same query always yields the same candidates (no RNG, no network). Different
    queries yield different candidates, and ``source_name`` lets tests simulate
    two distinct backends that should dedupe against each other.
    """

    def __init__(self, source_name: str = "fake") -> None:
        self.source_name = source_name

    def search(self, query: str, limit: int) -> list[Candidate]:
        if limit <= 0:
            return []
        out: list[Candidate] = []
        for i in range(limit):
            h = hashlib.sha256(f"{query}:{i}".encode("utf-8")).hexdigest()
            # DOI depends ONLY on (query, i) — NOT on source_name — so two fake
            # sources over the same query produce dedupe-able overlapping DOIs.
            doi = f"10.9999/fake.{h[:10]}"
            # Deterministic descending score in (0, 1].
            score = 1.0 - i * (0.5 / max(limit, 1))
            out.append(
                Candidate(
                    title=f"Fake paper {h[:6]} about {query}",
                    authors=[f"Author{i}, F."],
                    year=str(2000 + (int(h[:2], 16) % 25)),
                    doi=doi,
                    abstract_snippet=f"Deterministic abstract snippet {h[6:16]}.",
                    source=self.source_name,
                    score=round(score, 4),
                )
            )
        return out


# --- Real sources: Semantic Scholar + PubMed (stdlib urllib only) -----------

def _user_agent() -> str:
    """Build the discovery User-Agent string with the resolved contact email."""
    return f"scholia-discovery/0.1 (local; mailto:{_contact_email()})"


def _http_get_json(url: str, timeout: int = 15) -> dict:
    """GET a URL and parse JSON. Raises DiscoveryUnavailable on any network error."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError, ValueError) as exc:
        raise DiscoveryUnavailable(f"GET {url!r} failed: {exc}") from exc


def _snippet(text: str, n: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class SemanticScholarSource:
    """Real source querying the free Semantic Scholar Academic Graph API.

    READ-ONLY search. No API key required for modest use. Network is lazy and
    guarded — raises ``DiscoveryUnavailable`` when offline/rate-limited.
    """

    _ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
    source_name = "semanticscholar"

    def search(self, query: str, limit: int) -> list[Candidate]:
        if not query or limit <= 0:
            return []
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": min(int(limit), 100),
                "fields": "title,year,abstract,externalIds,authors",
            }
        )
        data = _http_get_json(f"{self._ENDPOINT}?{params}")
        papers = data.get("data") or []
        out: list[Candidate] = []
        n = max(len(papers), 1)
        for rank, p in enumerate(papers):
            ext = p.get("externalIds") or {}
            doi = str(ext.get("DOI") or "").strip()
            authors = [
                str(a.get("name", "")).strip()
                for a in (p.get("authors") or [])
                if a.get("name")
            ]
            # Rank-based descending score in (0, 1] (S2 returns by relevance).
            score = round(1.0 - rank / n, 4)
            out.append(
                Candidate(
                    title=str(p.get("title") or "").strip(),
                    authors=authors,
                    year=str(p.get("year") or "").strip(),
                    doi=doi,
                    abstract_snippet=_snippet(p.get("abstract") or ""),
                    source=self.source_name,
                    score=score,
                )
            )
        return out


class PubMedSource:
    """Real source querying NCBI PubMed via E-utilities (esearch + esummary).

    READ-ONLY search. Uses the standard library only. Network is lazy and
    guarded — raises ``DiscoveryUnavailable`` when offline/rate-limited.
    """

    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    _TOOL = "scholia-discovery"
    source_name = "pubmed"

    def search(self, query: str, limit: int) -> list[Candidate]:
        if not query or limit <= 0:
            return []
        # NCBI E-utilities etiquette wants a tool + contact email; the email is a
        # generic project default unless the user sets SCHOLIA_CONTACT_EMAIL.
        email = _contact_email()
        search_params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "term": query,
                "retmax": min(int(limit), 100),
                "retmode": "json",
                "sort": "relevance",
                "tool": self._TOOL,
                "email": email,
            }
        )
        sdata = _http_get_json(f"{self._ESEARCH}?{search_params}")
        idlist = (sdata.get("esearchresult") or {}).get("idlist") or []
        if not idlist:
            return []
        summary_params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "id": ",".join(idlist),
                "retmode": "json",
                "tool": self._TOOL,
                "email": email,
            }
        )
        summ = _http_get_json(f"{self._ESUMMARY}?{summary_params}")
        result = summ.get("result") or {}
        out: list[Candidate] = []
        n = max(len(idlist), 1)
        for rank, pmid in enumerate(idlist):
            rec = result.get(pmid)
            if not isinstance(rec, dict):
                continue
            doi = ""
            for aid in rec.get("articleids") or []:
                if str(aid.get("idtype", "")).lower() == "doi":
                    doi = str(aid.get("value", "")).strip()
                    break
            authors = [
                str(a.get("name", "")).strip()
                for a in (rec.get("authors") or [])
                if a.get("name")
            ]
            year = ""
            pubdate = str(rec.get("pubdate") or "").strip()
            if pubdate:
                year = pubdate.split(" ")[0]
            score = round(1.0 - rank / n, 4)
            out.append(
                Candidate(
                    title=str(rec.get("title") or "").strip(),
                    authors=authors,
                    year=year,
                    doi=doi,
                    abstract_snippet="",  # esummary has no abstract; keep it light
                    source=self.source_name,
                    score=score,
                )
            )
        return out
