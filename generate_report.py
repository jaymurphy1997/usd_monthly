#!/usr/bin/env python3
"""
USD Online Monthly Report Generator
====================================

Queries Google BigQuery (TESTUSDSession table), computes month-over-month (MoM)
and year-over-year (YoY) performance deltas, and renders a self-contained HTML
report (Chart.js + sortable tables + CSV export) from `report_template.html`.

Replaces the prior Looker Studio dashboard + Word document workflow for the
University of San Diego Online Graduate Programs.

Usage
-----
    pip install google-cloud-bigquery pandas
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # bash
    $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\path\\to\\service-account.json"  # PowerShell

    python generate_report.py --year 2026 --month 3 \\
        --project <GCP_PROJECT_ID> --dataset <BQ_DATASET>

Preview without BigQuery (generates realistic synthetic data):

    python generate_report.py --year 2026 --month 3 --mock

Output: reports/USD_Online_march_2026.html  (one file per month, never overwritten
without --force)
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from datetime import date, timedelta
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TABLE = "TESTUSDSession"
TEMPLATE_FILE = "report_template.html"
OUTPUT_DIR = "reports"

# HubSpot form-submission data (separate BigQuery dataset, same project).
HUBSPOT_DATASET = "USD_HubSpot_Data"
HUBSPOT_TABLE = "hubspot_submissions"          # one row per form submission
HUBSPOT_LOOKUP_TABLE = "hubspot_sales_cycle_lookup"  # form_name_hubspot -> sales_cycle

# Number of weeks of history rendered in the per-program trend line charts.
WEEK_WINDOW = 26

# Persisted history cache (lives under OUTPUT_DIR, which is git-ignored). Older
# months are reused from this file so repeat runs only re-query the current month.
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")

# The 15 programs included in the monthly report, in canonical display order.
# `code` is the value found in BigQuery (program_category / lead_program).
PROGRAMS: list[dict[str, str]] = [
    {"id": "informatics", "code": "Informatics", "name": "Health Care Informatics",                  "short": "Informatics"},
    {"id": "cyberops",    "code": "CyberOps",    "name": "Cyber Security Operations",                "short": "CyberOps"},
    {"id": "cybereng",    "code": "CyberEng",    "name": "Cyber Security Engineering",               "short": "CyberEng"},
    {"id": "med",         "code": "MED",         "name": "Master of Education",                      "short": "M.Ed."},
    {"id": "lepsl",       "code": "LEPSL",       "name": "Law Enforcement & Public Safety Leadership","short": "LEPSL"},
    {"id": "datascience", "code": "Data Science","name": "Applied Data Science",                     "short": "Data Sci"},
    {"id": "msaai",       "code": "MSAAI",       "name": "Applied Artificial Intelligence",          "short": "AAI"},
    {"id": "msldt",       "code": "MSLDT",       "name": "Learning Design & Technology",             "short": "LDT"},
    {"id": "mts",         "code": "MTS",         "name": "Theological Studies",                      "short": "MTS"},
    {"id": "mesh",        "code": "MESH",        "name": "Engineering for Sustainability & Health",  "short": "MESH"},
    {"id": "msha",        "code": "MSHA",        "name": "MS of Humanitarian Action",                "short": "MSHA"},
    {"id": "msnp",        "code": "MSNP",        "name": "Nonprofit Leadership & Management",         "short": "Nonprofit"},
    {"id": "eml",         "code": "EML",         "name": "Engineering Management & Leadership",      "short": "EML"},
    {"id": "msitl",       "code": "MSITL",       "name": "Information Technology Leadership",         "short": "ITL"},
    {"id": "msnnl",       "code": "MSNNL",       "name": "Nursing Leadership",                       "short": "Nursing"},
    {"id": "bsn",         "code": "BSN",         "name": "Bachelor of Science in Nursing",           "short": "BSN"},
]

PROGRAM_CODES = [p["code"] for p in PROGRAMS]
PROGRAM_ID_TO_CODE = {p["id"]: p["code"] for p in PROGRAMS}

# lead_program normalization: raw BigQuery value -> canonical program code.
LEAD_PROGRAM_NORMALIZATION = {
    "MSAAIS": "MSAAI",
    "LDT": "MSLDT",
    "Default - BSN": "BSN",
    "Default - BSN in Nursing": "BSN",
}

# Maps each report program `id` -> the HubSpot form-tag codes that belong to it.
# HubSpot encodes the program in the form_name bracket tag, e.g. "[PCE-MSHCI-ALL] …"
# or "[MSNNL] …". `hubspot_program_from_form` normalizes a tag to one of these codes.
# `CYBER` is intentionally absent: it is an ambiguous generic bucket that could be
# either CyberOps or CyberEng, so we leave those submissions unattributed.
HUBSPOT_PROGRAM_CODES: dict[str, list[str]] = {
    "informatics": ["MSHCI", "HCI"],
    "cyberops":    ["MSCSOL"],
    "cybereng":    ["MSCSE"],
    "med":         ["MED", "MEd"],
    "lepsl":       ["MSLEPSL", "LEPSL"],
    "datascience": ["MSADS"],
    "msaai":       ["MSAAI"],
    "msldt":       ["MSLDT", "LDT"],
    "mts":         ["MTS"],
    "mesh":        ["MSESH", "MESH"],
    "msha":        ["MSHA"],
    "msnp":        ["MSNP", "NP"],
    "eml":         ["MSEML", "EML"],
    "msitl":       ["MSITL"],
    "msnnl":       ["MSNNL"],
    "bsn":         ["BSN"],
}

# Reverse index: HubSpot tag code (upper) -> report program id.
_HUBSPOT_CODE_TO_ID = {
    code.upper(): pid for pid, codes in HUBSPOT_PROGRAM_CODES.items() for code in codes
}

HUBSPOT_UNCLASSIFIED = "Unclassified"

# Channels whose names begin with these prefixes are treated as noise and are
# excluded from program-level breakdowns and charts (matches prior analyst
# workflow which filtered "Unknown - ..." channels). School-wide totals still
# include every row, but these channels are collapsed for display.
EXCLUDED_CHANNEL_PREFIXES = ("Unknown",)

# Sales-cycle stage labels.
SALES_READY_LABEL = "Sales Ready"
EARLY_STAGE_LABEL = "Early Stage"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def month_bounds(year: int, month: int) -> tuple[str, str, int]:
    """Return (start_iso, end_iso, num_days) for the given year/month."""
    days = calendar.monthrange(year, month)[1]
    start = date(year, month, 1).isoformat()
    end = date(year, month, days).isoformat()
    return start, end, days


def prior_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def date_periods(year: int, month: int) -> dict[str, tuple[str, str]]:
    """Compute current, prior-month (MoM), and prior-year (YoY) date ranges."""
    cur_start, cur_end, _ = month_bounds(year, month)
    pm_year, pm_month = prior_month(year, month)
    pm_start, pm_end, _ = month_bounds(pm_year, pm_month)
    py_start, py_end, _ = month_bounds(year - 1, month)
    return {
        "current": (cur_start, cur_end),
        "prior_month": (pm_start, pm_end),
        "prior_year": (py_start, py_end),
    }


# ---------------------------------------------------------------------------
# BigQuery queries
# ---------------------------------------------------------------------------


def _fq_table(project: str, dataset: str) -> str:
    return f"`{project}.{dataset}.{TABLE}`"


def build_queries(project: str, dataset: str, start: str, end: str) -> dict[str, str]:
    """Build the 7 query strings for a single date period."""
    t = _fq_table(project, dataset)
    where = f"WHERE date BETWEEN '{start}' AND '{end}'"
    conv = f"{where}\n  AND conversion > 0"
    return {
        # Q1: Sessions by program + channel
        "sessions": f"""
SELECT program_category AS program, default_channel AS channel,
       SUM(session) AS sessions, SUM(page_view) AS pageviews
FROM {t}
{where}
GROUP BY 1, 2
""".strip(),
        # Q2: Conversions by program + channel
        "conversions": f"""
SELECT lead_program AS program, default_channel AS channel,
       SUM(conversion) AS conversions
FROM {t}
{conv}
GROUP BY 1, 2
""".strip(),
        # Q3: Sales cycle breakdown
        "sales_cycle": f"""
SELECT lead_program AS program, sales_cycle, SUM(conversion) AS conversions
FROM {t}
{conv}
GROUP BY 1, 2
""".strip(),
        # Q4: Landing page sessions
        "lp_sessions": f"""
SELECT program_category AS program, default_channel AS channel,
       landing_page_location AS landing_page, SUM(session) AS sessions
FROM {t}
{where}
GROUP BY 1, 2, 3
""".strip(),
        # Q5: Landing page conversions
        "lp_conversions": f"""
SELECT lead_program AS program, default_channel AS channel,
       landing_page_location AS landing_page, SUM(conversion) AS conversions
FROM {t}
{conv}
GROUP BY 1, 2, 3
""".strip(),
        # Q6: School-wide totals by channel
        "schoolwide": f"""
SELECT default_channel AS channel, SUM(session) AS sessions,
       SUM(page_view) AS pageviews, SUM(conversion) AS conversions
FROM {t}
{where}
GROUP BY 1
ORDER BY 2 DESC
""".strip(),
        # Q7: School-wide conversions by channel
        "schoolwide_conv": f"""
SELECT default_channel AS channel, SUM(conversion) AS conversions
FROM {t}
{conv}
GROUP BY 1
""".strip(),
    }


def run_period_queries(client, project: str, dataset: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Execute all 7 queries for one date period; return name -> DataFrame."""
    queries = build_queries(project, dataset, start, end)
    out: dict[str, pd.DataFrame] = {}
    for name, sql in queries.items():
        out[name] = client.query(sql).to_dataframe()
    return out


def make_client(project: str):
    """Create a BigQuery client (imported lazily so --mock needs no creds)."""
    from google.cloud import bigquery

    return bigquery.Client(project=project)


def fetch_all(project: str, dataset: str, year: int, month: int, client=None) -> dict[str, dict[str, pd.DataFrame]]:
    """Run every query for current / prior-month / prior-year periods."""
    if client is None:
        client = make_client(project)
    periods = date_periods(year, month)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for label, (start, end) in periods.items():
        print(f"  Querying {label}: {start} -> {end}")
        data[label] = run_period_queries(client, project, dataset, start, end)
    return data


# ---------------------------------------------------------------------------
# Time-series history (26-week trend charts)
# ---------------------------------------------------------------------------


def week_window(year: int, month: int, n: int = WEEK_WINDOW) -> tuple[str, str, list[str]]:
    """Return (win_start, win_end, week_keys) for the last n Monday-anchored ISO weeks
    ending at the report month-end. week_keys are ISO dates of each week's Monday."""
    _, month_end, _ = month_bounds(year, month)
    end_d = date.fromisoformat(month_end)
    end_monday = end_d - timedelta(days=end_d.weekday())  # Monday of the report-end week
    mondays = [end_monday - timedelta(weeks=i) for i in range(n)]
    mondays.reverse()
    week_keys = [d.isoformat() for d in mondays]
    return mondays[0].isoformat(), month_end, week_keys


def build_weekly_queries(project: str, dataset: str, start: str, end: str,
                         lp_top_n: int = 8) -> dict[str, str]:
    """Lean per-week aggregates for the trend charts (sessions/conv/landing pages).

    Program + channel filters are pushed into SQL and landing pages are capped to the
    top N per (program, week) server-side, so the whole window downloads small.
    """
    t = _fq_table(project, dataset)
    where = f"WHERE date BETWEEN '{start}' AND '{end}'"
    codes = ", ".join(f"'{c}'" for c in PROGRAM_CODES)
    wk = "DATE_TRUNC(date, WEEK(MONDAY))"
    # Matches is_excluded_channel: drop null channels and any 'Unknown...' bucket.
    keep_channel = "default_channel IS NOT NULL AND NOT STARTS_WITH(default_channel, 'Unknown')"
    return {
        "channels": f"""
SELECT program_category AS program, default_channel AS channel,
       {wk} AS week_start, SUM(session) AS sessions
FROM {t}
{where}
  AND program_category IN ({codes})
  AND {keep_channel}
GROUP BY 1, 2, 3
""".strip(),
        # Conversions stay program-unfiltered in SQL so lead_program variants
        # (MSAAIS, LDT, 'Default - BSN') survive for Python-side normalization.
        "conversions": f"""
SELECT lead_program AS program, default_channel AS channel,
       {wk} AS week_start, SUM(conversion) AS conversions
FROM {t}
{where}
  AND conversion > 0
  AND {keep_channel}
GROUP BY 1, 2, 3
""".strip(),
        "lp": f"""
SELECT program, week_start, landing_page, sessions FROM (
  SELECT program, week_start, landing_page, sessions,
         ROW_NUMBER() OVER (PARTITION BY program, week_start ORDER BY sessions DESC) AS rn
  FROM (
    SELECT program_category AS program, {wk} AS week_start,
           landing_page_location AS landing_page, SUM(session) AS sessions
    FROM {t}
    {where}
      AND program_category IN ({codes})
      AND {keep_channel}
    GROUP BY 1, 2, 3
  )
)
WHERE rn <= {lp_top_n}
""".strip(),
    }


def _reduce_programs(chan: pd.DataFrame, conv: pd.DataFrame, lp: pd.DataFrame,
                     lp_top_n: int = 8) -> dict[str, dict[str, Any]]:
    """Reduce one bucket's frames to a per-program-code record (sessions/conv/pages)."""
    out: dict[str, dict[str, Any]] = {}
    for code in PROGRAM_CODES:
        c = chan[chan["program"] == code] if not chan.empty else chan
        channels = (
            c.groupby("channel")["sessions"].sum().astype(int).to_dict() if not c.empty else {}
        )
        sessions = int(sum(channels.values()))
        conversions = int(conv.loc[conv["program"] == code, "conversions"].sum()) if not conv.empty else 0

        pages: dict[str, int] = {}
        if not lp.empty:
            lpp = lp[lp["program"] == code]
            if not lpp.empty:
                lpp = lpp.copy()
                lpp["url"] = lpp["landing_page"].map(strip_domain)
                pages = (
                    lpp.groupby("url")["sessions"].sum().astype(int)
                    .sort_values(ascending=False).head(lp_top_n).to_dict()
                )

        out[code] = {
            "sessions": sessions,
            "conversions": conversions,
            "convRate": conv_rate(conversions, sessions),
            "channels": channels,
            "landingPages": pages,
        }
    return out


def summarize_weekly(frames: dict[str, pd.DataFrame], week_keys: list[str]) -> dict[str, Any]:
    """Split windowed frames by week_start and reduce each week to per-program records."""
    chan = filter_program_channels(frames["channels"])
    conv = filter_program_channels(normalize_program(frames["conversions"]))
    lp = frames["lp"].copy()

    def wk_col(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["wk"] = pd.to_datetime(df["week_start"]).dt.strftime("%Y-%m-%d")
        return df

    chan, conv, lp = wk_col(chan), wk_col(conv), wk_col(lp)
    history: dict[str, Any] = {}
    for key in week_keys:
        cw = chan[chan["wk"] == key] if not chan.empty else chan
        vw = conv[conv["wk"] == key] if not conv.empty else conv
        lw = lp[lp["wk"] == key] if not lp.empty else lp
        history[key] = {"programs": _reduce_programs(cw, vw, lw)}
    return history


def load_history(path: str) -> dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  WARNING: could not read history file {path}; rebuilding.")
    return {}


def save_history(path: str, history: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=0)


def fetch_weekly_history(client, project: str, dataset: str, win_start: str, win_end: str,
                         week_keys: list[str], path: str) -> dict[str, Any]:
    """Query the full week window once (3 aggregated queries) and persist the snapshot."""
    print(f"  Weekly history: {win_start} -> {win_end} ({len(week_keys)} weeks)")
    frames = {name: client.query(sql).to_dataframe()
              for name, sql in build_weekly_queries(project, dataset, win_start, win_end).items()}
    history = summarize_weekly(frames, week_keys)
    save_history(path, history)
    return history


# ---------------------------------------------------------------------------
# HubSpot form submissions
# ---------------------------------------------------------------------------


def build_hubspot_query(project: str, hubspot_dataset: str, start: str, end: str) -> str:
    """Form-level submission counts for one period, with sales-cycle from the lookup."""
    subs = f"`{project}.{hubspot_dataset}.{HUBSPOT_TABLE}`"
    lookup = f"`{project}.{hubspot_dataset}.{HUBSPOT_LOOKUP_TABLE}`"
    return f"""
SELECT s.form_name AS form_name,
       COALESCE(l.sales_cycle, '{HUBSPOT_UNCLASSIFIED}') AS sales_cycle,
       COUNT(*) AS subs
FROM {subs} s
LEFT JOIN {lookup} l ON s.form_name = l.form_name_hubspot
WHERE DATE(s.date) BETWEEN '{start}' AND '{end}'
GROUP BY 1, 2
""".strip()


def fetch_hubspot_periods(client, project: str, hubspot_dataset: str, year: int, month: int
                          ) -> dict[str, pd.DataFrame]:
    """Query HubSpot form submissions for current / prior-month / prior-year periods."""
    periods = date_periods(year, month)
    out: dict[str, pd.DataFrame] = {}
    for label, (start, end) in periods.items():
        print(f"  HubSpot {label}: {start} -> {end}")
        df = client.query(build_hubspot_query(project, hubspot_dataset, start, end)).to_dataframe()
        if not df.empty:
            df = df.copy()
            df["pid"] = df["form_name"].map(hubspot_program_from_form)
            df["form"] = df["form_name"].map(clean_form_name)
        out[label] = df
    return out


def _hs_subset(df: pd.DataFrame, program_id: str) -> pd.DataFrame:
    if df.empty or "pid" not in df.columns:
        return pd.DataFrame(columns=["form", "sales_cycle", "subs"])
    return df[df["pid"] == program_id]


def build_program_hubspot(periods: dict[str, pd.DataFrame], program_id: str) -> dict[str, Any] | None:
    """Per-program HubSpot summary: totals, sales-cycle split, and form-level rows."""
    cur = _hs_subset(periods.get("current", pd.DataFrame()), program_id)
    pm = _hs_subset(periods.get("prior_month", pd.DataFrame()), program_id)
    py = _hs_subset(periods.get("prior_year", pd.DataFrame()), program_id)

    def total(df: pd.DataFrame) -> int:
        return int(df["subs"].sum()) if not df.empty else 0

    def cycle(df: pd.DataFrame, name: str) -> int:
        return int(df.loc[df["sales_cycle"] == name, "subs"].sum()) if not df.empty else 0

    def form_map(df: pd.DataFrame) -> dict[tuple[str, str], int]:
        if df.empty:
            return {}
        g = df.groupby(["form", "sales_cycle"], as_index=False)["subs"].sum()
        return {(r["form"], r["sales_cycle"]): int(r["subs"]) for _, r in g.iterrows()}

    cur_total = total(cur)
    if cur_total == 0 and total(pm) == 0 and total(py) == 0:
        return None  # program has no HubSpot footprint at all

    cur_forms, pm_forms, py_forms = form_map(cur), form_map(pm), form_map(py)
    forms: list[dict[str, Any]] = []
    for (form, sc), subs in sorted(cur_forms.items(), key=lambda kv: kv[1], reverse=True):
        forms.append({
            "form": form,
            "salesCycle": sc,
            "subs": subs,
            "momDelta": subs - pm_forms.get((form, sc), 0),
            "yoyDelta": subs - py_forms.get((form, sc), 0),
        })

    return {
        "total": cur_total,
        "momTotal": pct_change(cur_total, total(pm)),
        "yoyTotal": pct_change(cur_total, total(py)),
        "momTotalDelta": cur_total - total(pm),
        "yoyTotalDelta": cur_total - total(py),
        "salesReady": cycle(cur, SALES_READY_LABEL),
        "earlyStage": cycle(cur, EARLY_STAGE_LABEL),
        "unclassified": cycle(cur, HUBSPOT_UNCLASSIFIED),
        "forms": forms,
    }


# ---------------------------------------------------------------------------
# Normalization & filtering
# ---------------------------------------------------------------------------


def normalize_program(df: pd.DataFrame, col: str = "program") -> pd.DataFrame:
    """Apply lead_program normalization rules in place-safe fashion."""
    if df.empty:
        return df
    df = df.copy()
    df[col] = df[col].replace(LEAD_PROGRAM_NORMALIZATION)
    return df


def is_excluded_channel(channel: str | float) -> bool:
    if not isinstance(channel, str):
        return True
    return channel.startswith(EXCLUDED_CHANNEL_PREFIXES)


def filter_program_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 15 reportable programs and drop excluded (noise) channels."""
    if df.empty:
        return df
    mask = df["program"].isin(PROGRAM_CODES) & ~df["channel"].map(is_excluded_channel)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Delta math
# ---------------------------------------------------------------------------


def pct_change(current: float, prior: float) -> float | None:
    """Percent change, or None when prior is zero (division by zero)."""
    if prior in (0, 0.0) or prior is None:
        return None
    return round((current - prior) / prior * 100, 1)


def conv_rate(conversions: float, sessions: float) -> float | None:
    if not sessions:
        return None
    return round(conversions / sessions * 100, 2)


def strip_domain(url: str | float) -> str:
    """Reduce a full landing-page URL to its path (after sandiego.edu)."""
    if not isinstance(url, str) or not url:
        return "/"
    s = url
    for marker in ("sandiego.edu",):
        idx = s.find(marker)
        if idx != -1:
            s = s[idx + len(marker):]
            break
    else:
        # No domain marker; strip scheme + host if present.
        if "://" in s:
            s = "/" + s.split("://", 1)[1].split("/", 1)[-1]
    s = s.split("?")[0].split("#")[0]
    if not s.startswith("/"):
        s = "/" + s
    return s or "/"


# ---------------------------------------------------------------------------
# HubSpot form helpers
# ---------------------------------------------------------------------------


def hubspot_program_from_form(form_name: str | float) -> str | None:
    """Map a HubSpot form_name to a report program `id` via its bracket tag.

    Form names look like "[PCE-MSHCI-ALL] [GF] [RMI] PPC LP …" or "[MSNNL] [RMI] …".
    We take the leading bracket token, drop a `PCE-` prefix and trailing `-ALL`/`-OL`/
    `-OC` suffixes, then look the code up in HUBSPOT_PROGRAM_CODES. Returns None for
    forms with no recognizable program tag (e.g. school-wide `[PCE-ALL]`, `CYBER`).
    """
    if not isinstance(form_name, str) or "[" not in form_name:
        return None
    tag = form_name.split("[", 1)[1].split("]", 1)[0].strip().upper()
    if tag.startswith("PCE-"):
        tag = tag[4:]
    for suffix in ("-ALL", "-OL", "-OC"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    return _HUBSPOT_CODE_TO_ID.get(tag)


def clean_form_name(form_name: str | float) -> str:
    """Strip bracket tags and the boilerplate suffix for a readable form label."""
    if not isinstance(form_name, str) or not form_name:
        return "—"
    s = form_name
    # Remove all leading "[...]" tag groups.
    while s.lstrip().startswith("["):
        s = s.lstrip()
        end = s.find("]")
        if end == -1:
            break
        s = s[end + 1:]
    # Drop the "( Do not delete or edit )" style trailing note.
    idx = s.find("(")
    if idx != -1 and "delete" in s[idx:].lower():
        s = s[:idx]
    s = s.strip()
    return s or form_name.strip()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def program_session_totals(period: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sessions + pageviews per program (from program_category)."""
    df = filter_program_channels(period["sessions"])
    if df.empty:
        return pd.DataFrame(columns=["program", "sessions", "pageviews"])
    return df.groupby("program", as_index=False)[["sessions", "pageviews"]].sum()


def program_conv_totals(period: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Conversions per program (from lead_program, normalized)."""
    df = normalize_program(period["conversions"])
    df = filter_program_channels(df)
    if df.empty:
        return pd.DataFrame(columns=["program", "conversions"])
    return df.groupby("program", as_index=False)[["conversions"]].sum()


def lookup(df: pd.DataFrame, code: str, col: str) -> float:
    """Sum of `col` for a program code in an aggregated frame (0 if missing)."""
    if df.empty or "program" not in df.columns:
        return 0.0
    sel = df.loc[df["program"] == code, col]
    return float(sel.sum()) if len(sel) else 0.0


# ---------------------------------------------------------------------------
# JSON builders
# ---------------------------------------------------------------------------


SCHOOLWIDE_OTHER_CHANNEL = "Unknown / Other"


def build_schoolwide(data: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    """School-wide channel rows with MoM/YoY deltas. Includes ALL traffic, but
    collapses the many `Unknown - ...` channels into a single 'Unknown / Other'
    row so the totals stay complete while the table and charts stay readable."""

    def collapse(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        """Relabel excluded channels to 'Unknown / Other' and re-aggregate."""
        if df.empty:
            return df
        d = df.copy()
        d["channel"] = d["channel"].map(
            lambda c: SCHOOLWIDE_OTHER_CHANNEL if is_excluded_channel(c) else c
        )
        return d.groupby("channel", as_index=False)[value_cols].sum()

    cur = collapse(data["current"]["schoolwide"], ["sessions", "pageviews"])
    pm = collapse(data["prior_month"]["schoolwide"], ["sessions"])
    py = collapse(data["prior_year"]["schoolwide"], ["sessions"])
    cur_conv = collapse(data["current"]["schoolwide_conv"], ["conversions"])
    pm_conv = collapse(data["prior_month"]["schoolwide_conv"], ["conversions"])
    py_conv = collapse(data["prior_year"]["schoolwide_conv"], ["conversions"])

    def m(df: pd.DataFrame, channel: str, col: str) -> float:
        if df.empty or channel not in set(df["channel"]):
            return 0.0
        return float(df.loc[df["channel"] == channel, col].sum())

    rows: list[dict[str, Any]] = []
    for _, r in cur.iterrows():
        ch = r["channel"]
        sess = float(r["sessions"])
        pv = float(r.get("pageviews", 0) or 0)
        conv = m(cur_conv, ch, "conversions")
        pm_sess = m(pm, ch, "sessions")
        py_sess = m(py, ch, "sessions")
        pm_cv = m(pm_conv, ch, "conversions")
        py_cv = m(py_conv, ch, "conversions")
        rows.append({
            "channel": ch,
            "sessions": int(sess),
            "pageviews": int(pv),
            "conversions": int(conv),
            "convRate": conv_rate(conv, sess),
            "momSess": pct_change(sess, pm_sess),
            "yoySess": pct_change(sess, py_sess),
            "momSessDelta": int(sess - pm_sess),
            "yoySessDelta": int(sess - py_sess),
            "momConv": pct_change(conv, pm_cv),
            "yoyConv": pct_change(conv, py_cv),
            "momConvDelta": int(conv - pm_cv),
            "yoyConvDelta": int(conv - py_cv),
        })
    rows.sort(key=lambda x: x["sessions"], reverse=True)
    return rows


def build_program_channels(period: dict[str, pd.DataFrame], code: str) -> pd.DataFrame:
    """Per-channel sessions+conversions for one program (current period)."""
    sess = filter_program_channels(period["sessions"])
    sess = sess[sess["program"] == code]
    sess = sess.groupby("channel", as_index=False)[["sessions", "pageviews"]].sum() if not sess.empty \
        else pd.DataFrame(columns=["channel", "sessions", "pageviews"])

    conv = filter_program_channels(normalize_program(period["conversions"]))
    conv = conv[conv["program"] == code]
    conv = conv.groupby("channel", as_index=False)[["conversions"]].sum() if not conv.empty \
        else pd.DataFrame(columns=["channel", "conversions"])

    merged = pd.merge(sess, conv, on="channel", how="outer").fillna(0)
    return merged


def build_landing_pages(data: dict[str, dict[str, pd.DataFrame]], code: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Top-N landing pages for a program by current-month sessions."""
    cur = data["current"]
    pm = data["prior_month"]

    def lp_sessions(period: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = period["lp_sessions"]
        if df.empty:
            return pd.DataFrame(columns=["channel", "landing_page", "sessions"])
        df = df[df["program"] == code]
        df = df[~df["channel"].map(is_excluded_channel)]
        if df.empty:
            return pd.DataFrame(columns=["channel", "landing_page", "sessions"])
        return df.groupby(["channel", "landing_page"], as_index=False)[["sessions"]].sum()

    def lp_conv(period: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = normalize_program(period["lp_conversions"])
        if df.empty:
            return pd.DataFrame(columns=["channel", "landing_page", "conversions"])
        df = df[df["program"] == code]
        df = df[~df["channel"].map(is_excluded_channel)]
        if df.empty:
            return pd.DataFrame(columns=["channel", "landing_page", "conversions"])
        return df.groupby(["channel", "landing_page"], as_index=False)[["conversions"]].sum()

    cur_s = lp_sessions(cur)
    if cur_s.empty:
        return []
    cur_c = lp_conv(cur)
    pm_s = lp_sessions(pm)
    pm_c = lp_conv(pm)

    merged = pd.merge(cur_s, cur_c, on=["channel", "landing_page"], how="left").fillna({"conversions": 0})
    merged = merged.sort_values("sessions", ascending=False).head(top_n)

    def prior(df: pd.DataFrame, ch: str, lp: str, col: str) -> float:
        if df.empty:
            return 0.0
        sel = df[(df["channel"] == ch) & (df["landing_page"] == lp)]
        return float(sel[col].sum()) if len(sel) else 0.0

    pages: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        ch, lp = r["channel"], r["landing_page"]
        sess = float(r["sessions"])
        conv = float(r["conversions"])
        pages.append({
            "channel": ch,
            "url": strip_domain(lp),
            "sessions": int(sess),
            "momSess": int(sess - prior(pm_s, ch, lp, "sessions")),
            "conversions": int(conv),
            "momConv": int(conv - prior(pm_c, ch, lp, "conversions")),
        })
    return pages


def build_sales_cycle(period: dict[str, pd.DataFrame], code: str) -> tuple[int, int]:
    """Return (sales_ready, early_stage) conversion counts for a program."""
    df = normalize_program(period["sales_cycle"])
    if df.empty:
        return 0, 0
    df = df[df["program"] == code]
    ready = int(df.loc[df["sales_cycle"] == SALES_READY_LABEL, "conversions"].sum())
    early = int(df.loc[df["sales_cycle"] == EARLY_STAGE_LABEL, "conversions"].sum())
    return ready, early


def build_program_trend(history: dict[str, Any], trend_keys: list[str], code: str,
                        max_channels: int = 6, max_pages: int = 5) -> dict[str, Any]:
    """26-week trend series for one program: sessions-by-channel, conv rate, landing pages.

    trend_keys are ISO week-start dates ('YYYY-MM-DD'); labels are 'M/D' week starts.
    """
    labels = [f"{int(k.split('-')[1])}/{int(k.split('-')[2])}" for k in trend_keys]
    recs = [history.get(k, {}).get("programs", {}).get(code, {}) for k in trend_keys]

    # Sessions by channel: pick the channels with the most total sessions across the window.
    chan_totals: dict[str, int] = {}
    for r in recs:
        for ch, v in (r.get("channels") or {}).items():
            chan_totals[ch] = chan_totals.get(ch, 0) + int(v)
    top_channels = [ch for ch, _ in sorted(chan_totals.items(), key=lambda kv: kv[1], reverse=True)][:max_channels]
    chan_series = {ch: [int((r.get("channels") or {}).get(ch, 0)) for r in recs] for ch in top_channels}

    # Conversion rate trend (may contain nulls where a month had no sessions).
    conv_rate_series = [r.get("convRate") if r else None for r in recs]

    # Landing pages: top pages by the latest month's sessions.
    latest_pages = recs[-1].get("landingPages") if recs and recs[-1] else None
    latest_pages = latest_pages or {}
    top_pages = [u for u, _ in sorted(latest_pages.items(), key=lambda kv: kv[1], reverse=True)][:max_pages]
    page_series = {u: [int((r.get("landingPages") or {}).get(u, 0)) for r in recs] for u in top_pages}

    return {
        "months": labels,
        "channels": {"labels": labels, "series": chan_series},
        "convRate": conv_rate_series,
        "landingPages": {"series": page_series},
    }


def build_programs(data: dict[str, dict[str, pd.DataFrame]],
                   history: dict[str, Any] | None = None,
                   trend_keys: list[str] | None = None,
                   hubspot_periods: dict[str, pd.DataFrame] | None = None) -> list[dict[str, Any]]:
    """Build the per-program JSON array (all programs, canonical order)."""
    history = history or {}
    trend_keys = trend_keys or []
    cur_sess = program_session_totals(data["current"])
    pm_sess = program_session_totals(data["prior_month"])
    py_sess = program_session_totals(data["prior_year"])
    cur_conv = program_conv_totals(data["current"])
    pm_conv = program_conv_totals(data["prior_month"])
    py_conv = program_conv_totals(data["prior_year"])

    programs: list[dict[str, Any]] = []
    for p in PROGRAMS:
        code = p["code"]
        sessions = lookup(cur_sess, code, "sessions")
        pageviews = lookup(cur_sess, code, "pageviews")
        conversions = lookup(cur_conv, code, "conversions")

        pm_s = lookup(pm_sess, code, "sessions")
        py_s = lookup(py_sess, code, "sessions")
        pm_c = lookup(pm_conv, code, "conversions")
        py_c = lookup(py_conv, code, "conversions")

        cr = conv_rate(conversions, sessions)
        pm_cr = conv_rate(pm_c, pm_s)
        mom_cr = pct_change(cr, pm_cr) if (cr is not None and pm_cr) else None

        ready, early = build_sales_cycle(data["current"], code)

        chan = build_program_channels(data["current"], code)
        chan = chan.sort_values("sessions", ascending=False)
        top_channel = str(chan.iloc[0]["channel"]) if not chan.empty else "—"

        # Prior-month channel sessions/conversions for channel-detail deltas.
        pm_chan = build_program_channels(data["prior_month"], code)

        def pm_chan_val(ch: str, col: str) -> float:
            if pm_chan.empty:
                return 0.0
            sel = pm_chan.loc[pm_chan["channel"] == ch, col]
            return float(sel.sum()) if len(sel) else 0.0

        channel_details = []
        for _, r in chan.iterrows():
            ch = str(r["channel"])
            s = float(r["sessions"])
            c = float(r["conversions"])
            channel_details.append({
                "channel": ch,
                "sessions": int(s),
                "conversions": int(c),
                "momSessDelta": int(s - pm_chan_val(ch, "sessions")),
                "momConvDelta": int(c - pm_chan_val(ch, "conversions")),
            })

        programs.append({
            "id": p["id"],
            "code": code,
            "name": p["name"],
            "short": p["short"],
            "sessions": int(sessions),
            "pageviews": int(pageviews),
            "conversions": int(conversions),
            "convRate": cr,
            "salesReady": ready,
            "earlyStage": early,
            "momSess": pct_change(sessions, pm_s),
            "yoySess": pct_change(sessions, py_s),
            "momConv": pct_change(conversions, pm_c),
            "yoyConv": pct_change(conversions, py_c),
            "momCR": mom_cr,
            "topChannel": top_channel,
            "channels": {
                "labels": [d["channel"] for d in channel_details],
                "sessions": [d["sessions"] for d in channel_details],
                "conversions": [d["conversions"] for d in channel_details],
            },
            "channelDetails": channel_details,
            "landingPages": build_landing_pages(data, code),
            "trend": build_program_trend(history, trend_keys, code),
            "hubspot": build_program_hubspot(hubspot_periods or {}, p["id"]),
        })
    return programs


# ---------------------------------------------------------------------------
# Mock data (for previewing the template without BigQuery access)
# ---------------------------------------------------------------------------


def _mock_period_frames(seed_scale: float, year: int, month: int) -> dict[str, pd.DataFrame]:
    """Deterministic synthetic data for one period. seed_scale scales magnitudes."""
    channels = ["Organic Search", "Direct", "Paid Search", "Paid Social", "Email", "Referral", "Display"]
    # Deterministic per-program/channel weights (no RNG -> reproducible output).
    sess_rows, conv_rows, sc_rows = [], [], []
    lp_sess_rows, lp_conv_rows = [], []
    for pi, p in enumerate(PROGRAMS):
        code = p["code"]
        base = (700 + pi * 230)
        for ci, ch in enumerate(channels):
            weight = max(0.05, 1.0 - ci * 0.16)
            sessions = int(base * weight * seed_scale)
            pageviews = int(sessions * 1.15)
            conversions = int(sessions * (0.004 + (ci % 3) * 0.006))
            if sessions <= 0:
                continue
            sess_rows.append({"program": code, "channel": ch, "sessions": sessions, "pageviews": pageviews})
            if conversions > 0:
                conv_rows.append({"program": code, "channel": ch, "conversions": conversions})
            # Landing pages (2 per channel)
            for li in range(2):
                lp = f"https://www.sandiego.edu/online/{code.lower().replace(' ', '-')}/page-{li}/"
                lps = int(sessions * (0.6 if li == 0 else 0.4))
                lpc = int(conversions * (0.6 if li == 0 else 0.4))
                lp_sess_rows.append({"program": code, "channel": ch, "landing_page": lp, "sessions": lps})
                if lpc > 0:
                    lp_conv_rows.append({"program": code, "channel": ch, "landing_page": lp, "conversions": lpc})
        total_conv = int(base * 0.05 * seed_scale)
        sc_rows.append({"program": code, "sales_cycle": SALES_READY_LABEL, "conversions": int(total_conv * 0.9)})
        sc_rows.append({"program": code, "sales_cycle": EARLY_STAGE_LABEL, "conversions": int(total_conv * 0.1)})

    sess_df = pd.DataFrame(sess_rows)
    conv_df = pd.DataFrame(conv_rows)
    sw = sess_df.groupby("channel", as_index=False)[["sessions", "pageviews"]].sum()
    sw_conv = conv_df.groupby("channel", as_index=False)[["conversions"]].sum()
    sw = pd.merge(sw, sw_conv, on="channel", how="left").fillna(0)
    sw = sw.sort_values("sessions", ascending=False)
    return {
        "sessions": sess_df,
        "conversions": conv_df,
        "sales_cycle": pd.DataFrame(sc_rows),
        "lp_sessions": pd.DataFrame(lp_sess_rows),
        "lp_conversions": pd.DataFrame(lp_conv_rows),
        "schoolwide": sw[["channel", "sessions", "pageviews", "conversions"]],
        "schoolwide_conv": sw_conv,
    }


def mock_data(year: int, month: int) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        "current": _mock_period_frames(1.00, year, month),
        "prior_month": _mock_period_frames(0.92, *prior_month(year, month)),
        "prior_year": _mock_period_frames(0.84, year - 1, month),
    }


def mock_weekly_history(year: int, month: int, n: int = WEEK_WINDOW) -> tuple[dict[str, Any], list[str]]:
    """Synthetic 26-week history so --mock previews the weekly trend charts offline."""
    _, _, week_keys = week_window(year, month, n)
    history: dict[str, Any] = {}
    for i, key in enumerate(week_keys):
        # Gentle ramp + light wiggle so the weekly lines visibly move.
        scale = (0.70 + 0.012 * i) * (1.0 + 0.06 * ((i % 3) - 1)) / 4.3
        frames = _mock_period_frames(scale, year, month)
        chan = frames["sessions"][["program", "channel", "sessions"]]
        conv = frames["conversions"]
        lp = frames["lp_sessions"]
        history[key] = {"programs": _reduce_programs(
            filter_program_channels(chan),
            filter_program_channels(normalize_program(conv)),
            lp,
        )}
    return history, week_keys


def mock_hubspot_periods(year: int, month: int) -> dict[str, pd.DataFrame]:
    """Synthetic HubSpot form submissions for current/prior-month/prior-year."""
    forms = [
        ("[PCE-{tag}] [GF] [RMI] PPC LP ( Do not delete or edit )", SALES_READY_LABEL, 1.0),
        ("[PCE-{tag}] [GF] [RMI] Program Page RFI ( Do not delete or edit )", SALES_READY_LABEL, 0.4),
        ("[PCE-{tag}] [GF] [RESOURCE] Career Guide ( Do not delete or edit )", EARLY_STAGE_LABEL, 0.25),
    ]
    scales = {"current": 1.0, "prior_month": 0.85, "prior_year": 0.7}
    out: dict[str, pd.DataFrame] = {}
    for label, sc in scales.items():
        rows = []
        for pi, p in enumerate(PROGRAMS):
            codes = HUBSPOT_PROGRAM_CODES.get(p["id"]) or []
            if not codes:
                continue
            tag = codes[0]
            base = 30 + pi * 4
            for fname, cycle, w in forms:
                subs = int(base * w * sc)
                if subs <= 0:
                    continue
                rows.append({"form_name": fname.format(tag=tag), "sales_cycle": cycle, "subs": subs})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["pid"] = df["form_name"].map(hubspot_program_from_form)
            df["form"] = df["form_name"].map(clean_form_name)
        out[label] = df
    return out


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render(template_path: str, replacements: dict[str, str]) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    for key, value in replacements.items():
        html = html.replace("{{" + key + "}}", value)
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the USD Online monthly HTML report.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--project", help="GCP project ID (required unless --mock)")
    parser.add_argument("--dataset", help="BigQuery dataset (required unless --mock)")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data instead of BigQuery")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--template", default=TEMPLATE_FILE)
    parser.add_argument("--report-date", help="Report date label, e.g. 'March 12, 2026' (default: today)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing report file")
    parser.add_argument("--hubspot-dataset", default=HUBSPOT_DATASET,
                        help=f"BigQuery dataset for HubSpot tables (default: {HUBSPOT_DATASET})")
    parser.add_argument("--history-file", default=HISTORY_FILE,
                        help=f"Persisted trend-history cache (default: {HISTORY_FILE})")
    parser.add_argument("--trend-weeks", type=int, default=WEEK_WINDOW,
                        help=f"Weeks shown in the trend line charts (default: {WEEK_WINDOW})")
    parser.add_argument("--firebase-config",
                        help="Path to a Firebase web-config JSON to enable shared comments "
                             "(Firestore + Google sign-in). Omit for localStorage-only comments.")
    args = parser.parse_args()

    if not args.mock and (not args.project or not args.dataset):
        parser.error("--project and --dataset are required unless --mock is set")

    year, month = args.year, args.month
    start, end, days = month_bounds(year, month)
    month_name = calendar.month_name[month]

    # Acquire data.
    if args.mock:
        print(f"Generating MOCK data for {month_name} {year} (no BigQuery)...")
        data = mock_data(year, month)
        history, trend_keys = mock_weekly_history(year, month, args.trend_weeks)
        hubspot_periods = mock_hubspot_periods(year, month)
    else:
        print(f"Querying BigQuery for {month_name} {year}...")
        client = make_client(args.project)
        data = fetch_all(args.project, args.dataset, year, month, client=client)
        print("Querying weekly trend history...")
        win_start, win_end, trend_keys = week_window(year, month, args.trend_weeks)
        history = fetch_weekly_history(client, args.project, args.dataset,
                                       win_start, win_end, trend_keys, args.history_file)
        print("Querying HubSpot form submissions...")
        hubspot_periods = fetch_hubspot_periods(client, args.project, args.hubspot_dataset, year, month)

    print("Building report data structures...")
    programs = build_programs(data, history=history, trend_keys=trend_keys,
                              hubspot_periods=hubspot_periods)
    schoolwide = build_schoolwide(data)

    # Report date label.
    if args.report_date:
        report_date = args.report_date
    else:
        report_date = date.today().strftime("%B %d, %Y")

    # Comments: stable report id + optional Firebase config (None -> localStorage fallback).
    report_id = f"usd_online_{year}_{month:02d}"
    firebase_config: Any = None
    if args.firebase_config:
        if os.path.exists(args.firebase_config):
            with open(args.firebase_config, "r", encoding="utf-8") as f:
                firebase_config = json.load(f)
            print(f"  Comments: Firestore + Google sign-in enabled (config {args.firebase_config})")
        else:
            print(f"  WARNING: --firebase-config not found: {args.firebase_config}; "
                  "comments fall back to localStorage.", file=sys.stderr)

    replacements = {
        "MONTH_NAME": month_name,
        "YEAR": str(year),
        "MONTH_NUM": f"{month:02d}",
        "MONTH_DAYS": str(days),
        "MONTH_LC": month_name.lower(),
        "REPORT_DATE": report_date,
        "REPORT_ID": report_id,
        "PROGRAMS_JSON": json.dumps(programs, ensure_ascii=False),
        "SCHOOL_WIDE_JSON": json.dumps(schoolwide, ensure_ascii=False),
        "FIREBASE_CONFIG_JSON": json.dumps(firebase_config, ensure_ascii=False),
    }

    if not os.path.exists(args.template):
        print(f"ERROR: template not found: {args.template}", file=sys.stderr)
        return 1

    html = render(args.template, replacements)

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"USD_Online_{month_name.lower()}_{year}.html"
    out_path = os.path.join(args.output_dir, out_name)
    if os.path.exists(out_path) and not args.force:
        print(f"ERROR: {out_path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written: {out_path}")
    print(f"  Programs: {len(programs)}  School-wide channels: {len(schoolwide)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
