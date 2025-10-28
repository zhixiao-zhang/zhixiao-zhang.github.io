# build_yaml.py
#
# Pipeline:
# 1. load config + state
# 2. fetch recent arXiv candidates
# 3. incremental window filter (last_run -> now, seen_ids remove the redundancy)
# 4. DeepSeek relevance scoring (ai_score / ai_reason)
#    - keep only papers with ai_score >= cfg["ai_min_score"]
# 5. DeepSeek TL;DR for kept papers
# 6. write YAML snapshot for Hugo
# 7. update state.json
#
# Requirements:
#   pip install requests feedparser pyyaml openai
#
# Environment:
#   export DEEPSEEK_API_KEY="sk-...."
#
# Notes:
# - We DO NOT keep the old keyword score. AI score is now the "gate".
# - We still use incremental time window + seen_ids so we don't repeat.
#
# (c) zhixiao radar :)

import os
import json
import yaml
import textwrap
from typing import List, Dict, Tuple
from datetime import datetime, timezone, timedelta

from openai import OpenAI
from arxiv_fetch import query_arxiv


STATE_PATH = "state.json"


def debug(msg: str):
    print(f"[DEBUG] {msg}")


# ---------- config / state / time helpers ----------

def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # fill some sane defaults if missing
    cfg.setdefault("recent_days", 3)
    cfg.setdefault("max_results", 80)
    cfg.setdefault("output_dir", "data/arxiv")
    # new: required AI relevance threshold
    cfg.setdefault("ai_min_score", 7)
    return cfg


def parse_rfc3339(ts: str) -> datetime:
    """Parse timestamps like '2025-10-28T01:23:45Z' into aware UTC datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def load_state() -> Dict:
    """Load or init state.json."""
    if not os.path.exists(STATE_PATH):
        debug("No state.json found. Initializing new state.")
        return {
            "last_run_utc": None,
            "seen_ids": []
        }
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    debug(f"Loaded state: {state}")
    return state


def save_state(state: Dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    debug(f"Saved updated state: {state}")


# ---------- DeepSeek helpers ----------

def get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    return client


def llm_judge_relevance(client: OpenAI, paper: Dict, cfg: Dict) -> Dict:
    """
    Ask DeepSeek: is this paper relevant to our research focus?
    Returns:
      {
        "relevance": number 0-10,
        "reason": "...short justification..."
      }
    If anything fails, fallback to low score so it gets filtered out.
    """

    if client is None:
        # no model available -> treat as irrelevant but don't crash
        return {"relevance": 0, "reason": "no DEEPSEEK_API_KEY"}

    focus = "The researcher focuses on:\n" + "\n".join(
        [f"- {kw}" for kw in cfg.get("must_include", [])]
    )

    user_prompt = textwrap.dedent(f"""
    {focus}

    Evaluate relevance of this paper to the focus above.
    Title: {paper.get("title")}
    Abstract: {paper.get("summary")}

    Respond ONLY as strict JSON, no markdown, no commentary. Format:

    {{
      "relevance": <0-10 number>,
      "reason": "<one-sentence justification>"
    }}
    """)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert systems security / PL assistant. Always follow instructions exactly."},
                {"role": "user", "content": focus + "\n\n" + user_prompt},
            ],
            temperature=1.3,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        # normalize output
        return {
            "relevance": data.get("relevance", 0),
            "reason": data.get("reason", "").strip()
        }
    except Exception as e:
        print(f"[WARN] relevance check failed for {paper.get('arxiv_id')}: {e}")
        return {"relevance": 0, "reason": "parse error or API failure"}


def llm_tldr(client: OpenAI, paper: Dict) -> str:
    """
    Ask DeepSeek to summarize the paper in a TL;DR form.
    Returns short bullet-y string. Fail soft.
    """

    if client is None:
        return ""

    prompt = textwrap.dedent(f"""
    You are assisting a systems security and programming-languages researcher.

    Summarize the following arXiv paper in 3–5 short bullet points.
    Focus on:
    1. The problem addressed (what real risk / gap?).
    2. The core technical method or mechanism.
    3. Why it matters for memory safety, system security, or low-level safety.

    Be concrete. Avoid filler like "novel approach" or "significant improvement".
    Output plain text bullets starting with "- ".

    Title: {paper.get("title")}
    Abstract:
    {paper.get("summary")}

    TL;DR:
    """)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a concise, domain-aware summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.3,
            max_tokens=300,
        )
        tldr = resp.choices[0].message.content.strip()
        # clean common leading indentation
        tldr = textwrap.dedent(tldr).strip()
        return tldr
    except Exception as e:
        print(f"[WARN] TL;DR failed for {paper.get('arxiv_id')}: {e}")
        return ""


# ---------- filtering logic before AI ----------

def incremental_window_filter(
    papers: List[Dict],
    last_run_dt: datetime,
    now_dt: datetime,
    seen_ids: List[str],
) -> Tuple[List[Dict], List[str]]:
    """
    Basic pre-filter before AI:
    - Only keep papers updated in (last_run_dt, now_dt]
    - Drop already-seen arxiv_id versions
    This step does NOT do keyword scoring anymore.
    We just do time + dedup to get "candidates for AI".
    """

    seen_set = set(seen_ids)
    kept: List[Dict] = []

    for p in papers:
        if "updated" not in p or "arxiv_id" not in p:
            continue

        updated_dt = parse_rfc3339(p["updated"])

        # incremental window
        if not (last_run_dt < updated_dt <= now_dt):
            continue

        # dedup by versioned arxiv_id (v1, v2...)
        if p["arxiv_id"] in seen_set:
            continue

        kept.append(p)
        seen_set.add(p["arxiv_id"])

    return kept, list(seen_set)


# ---------- YAML builder ----------

def build_output_structure(
    papers: List[Dict],
    cfg: Dict,
    generated_at: datetime
) -> Dict:
    """
    Prepare final YAML for Hugo.
    """
    out = {
        "title": cfg.get("site_title", "ArXiv Radar"),
        "description": cfg.get("site_description", ""),
        "generated_at": generated_at.isoformat(),
        "papers": [],
    }

    for p in papers:
        out["papers"].append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "authors": p.get("authors", []),
            "summary": p.get("summary"),
            "tldr": p.get("tldr", ""),
            "ai_score": p.get("ai_score", 0),
            "ai_reason": p.get("ai_reason", ""),
            "updated": p.get("updated"),
            "published": p.get("published"),
            "pdf_url": p.get("pdf_url"),
            "primary_category": p.get("primary_category"),
            "categories": p.get("categories", []),
        })
    return out


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# ---------- main ----------

def main():
    debug("=== radar run start ===")

    cfg = load_config()
    state = load_state()

    now_dt = datetime.now(timezone.utc)
    debug(f"Now UTC: {now_dt.isoformat()}")

    # compute last_run_dt
    if state["last_run_utc"] is None:
        # first run fallback: recent_days window
        last_run_dt = now_dt - timedelta(days=cfg["recent_days"])
        debug(f"First run: using fallback last_run_dt={last_run_dt.isoformat()}")
    else:
        last_run_dt = datetime.fromisoformat(state["last_run_utc"])
        debug(f"Using last_run_dt from state: {last_run_dt.isoformat()}")

    # 1. Fetch raw candidates from arXiv
    debug("Querying arXiv...")
    raw_papers = query_arxiv(
        categories=cfg["categories"],
        max_results=cfg.get("max_results", 80),
    )
    debug(f"Fetched {len(raw_papers)} raw papers")

    # 2. Incremental window filter + dedup
    candidates, updated_seen_ids = incremental_window_filter(
        papers=raw_papers,
        last_run_dt=last_run_dt,
        now_dt=now_dt,
        seen_ids=state.get("seen_ids", []),
    )
    debug(f"{len(candidates)} papers after time-window & dedup pre-filter")

    # 3. AI relevance check
    client = get_deepseek_client()
    ai_filtered: List[Dict] = []
    for p in candidates:
        res = llm_judge_relevance(client, p, cfg)
        p["ai_score"] = res.get("relevance", 0)
        p["ai_reason"] = res.get("reason", "")
        debug(f"AI score {p['ai_score']} for {p.get('arxiv_id')} : {p['ai_reason']}")
        if p["ai_score"] >= cfg["ai_min_score"]:
            ai_filtered.append(p)

    debug(f"{len(ai_filtered)} papers kept after AI relevance filter (threshold {cfg['ai_min_score']})")

    # 4. Generate TL;DR for the kept ones
    for p in ai_filtered:
        p["tldr"] = llm_tldr(client, p)

    # 5. Build YAML doc
    final_doc = build_output_structure(
        papers=ai_filtered,
        cfg=cfg,
        generated_at=now_dt,
    )

    # 6. Write snapshot YAML
    today_str = now_dt.date().isoformat()
    out_dir = cfg.get("output_dir", "data/arxiv")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{today_str}.yaml")

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(final_doc, f, sort_keys=False, allow_unicode=True)

    debug(f"Wrote {out_path} with {len(final_doc['papers'])} AI-approved papers.")

    # 7. Update and save state
    state["last_run_utc"] = now_dt.isoformat()
    state["seen_ids"] = updated_seen_ids
    save_state(state)

    debug("=== radar run end ===")


if __name__ == "__main__":
    main()
