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
from datetime import date
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TABLE = "TESTUSDSession"
TEMPLATE_FILE = "report_template.html"
OUTPUT_DIR = "reports"

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
]

PROGRAM_CODES = [p["code"] for p in PROGRAMS]

# lead_program normalization: raw BigQuery value -> canonical program code.
LEAD_PROGRAM_NORMALIZATION = {
    "MSAAIS": "MSAAI",
    "LDT": "MSLDT",
}

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


def fetch_all(project: str, dataset: str, year: int, month: int) -> dict[str, dict[str, pd.DataFrame]]:
    """Run every query for current / prior-month / prior-year periods."""
    from google.cloud import bigquery  # imported lazily so --mock needs no creds

    client = bigquery.Client(project=project)
    periods = date_periods(year, month)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for label, (start, end) in periods.items():
        print(f"  Querying {label}: {start} -> {end}")
        data[label] = run_period_queries(client, project, dataset, start, end)
    return data


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


def build_programs(data: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    """Build the per-program JSON array (all 15 programs, canonical order)."""
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
    else:
        print(f"Querying BigQuery for {month_name} {year}...")
        data = fetch_all(args.project, args.dataset, year, month)

    print("Building report data structures...")
    programs = build_programs(data)
    schoolwide = build_schoolwide(data)

    # Report date label.
    if args.report_date:
        report_date = args.report_date
    else:
        report_date = date.today().strftime("%B %d, %Y")

    replacements = {
        "MONTH_NAME": month_name,
        "YEAR": str(year),
        "MONTH_NUM": f"{month:02d}",
        "MONTH_DAYS": str(days),
        "MONTH_LC": month_name.lower(),
        "REPORT_DATE": report_date,
        "PROGRAMS_JSON": json.dumps(programs, ensure_ascii=False),
        "SCHOOL_WIDE_JSON": json.dumps(schoolwide, ensure_ascii=False),
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
