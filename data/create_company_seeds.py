"""
One-time script to create the company_seeds.parquet seed file.
Run: python data/create_company_seeds.py
"""
from __future__ import annotations
from pathlib import Path
import polars as pl

# Known penny stock universe with manually-assessed reality attributes.
# Reality data sourced from SEC EDGAR, LinkedIn, company websites.
SEEDS = [
    # ticker, name, exchange, sector, country_iso3,
    # employee_count, location_count, location_type,
    # named_customers_count, has_shipped_product,
    # auditor, auditor_tier, years_operating, ceo_verifiable
    ("GTII",  "Global Arena Holding",       "OTC",    "Other",    "USA", 0,   1, "registered_agent", 0,  False, "unknown",            "unknown", 0.5,  False),
    ("LGVN",  "Longeviti Neuro Solutions",  "NASDAQ", "Biotech",  "USA", 12,  1, "real_office",      2,  True,  "Marcum LLP",         "mid",     4.0,  True),
    ("SOPA",  "Society Pass",               "NASDAQ", "Tech",     "USA", 80,  3, "multi_site",       5,  True,  "Assurance Dim.",     "micro",   5.0,  True),
    ("SHOT",  "Safety Shot",                "NASDAQ", "Cannabis", "USA", 8,   1, "real_office",      1,  True,  "Salberg & Co",       "micro",   3.0,  True),
    ("GFAI",  "Guardforce AI",              "NASDAQ", "Tech",     "USA", 900, 8, "multi_site",       15, True,  "Marcum Bernstein",   "mid",     8.0,  True),
    ("VERB",  "Verb Technology",            "NASDAQ", "Tech",     "USA", 25,  1, "real_office",      3,  True,  "Weinberg & Company", "micro",   7.0,  True),
    ("FFIE",  "Faraday Future",             "NASDAQ", "EV",       "USA", 400, 2, "multi_site",       1,  False, "Deloitte",           "big4",    8.0,  True),
    ("MULN",  "Mullen Automotive",          "NASDAQ", "EV",       "USA", 110, 3, "multi_site",       2,  False, "Weinberg & Company", "micro",   5.0,  True),
    ("MMAT",  "Meta Materials",             "NASDAQ", "Tech",     "USA", 150, 2, "multi_site",       4,  True,  "KPMG",               "big4",    6.0,  True),
    ("PROG",  "Progenity",                  "NASDAQ", "Biotech",  "USA", 200, 2, "multi_site",       0,  True,  "Ernst & Young",      "big4",    8.0,  True),
    ("NKLA",  "Nikola Corporation",         "NASDAQ", "EV",       "USA", 450, 2, "multi_site",       1,  False, "Ernst & Young",      "big4",    7.0,  True),
    ("CLOV",  "Clover Health",              "NASDAQ", "Financial","USA", 900, 3, "multi_site",       20, True,  "Deloitte",           "big4",    7.0,  True),
    ("WKHS",  "Workhorse Group",            "NASDAQ", "EV",       "USA", 120, 2, "multi_site",       2,  True,  "Clark Nuber",        "mid",     8.0,  True),
    ("IDEX",  "Ideanomics",                 "NASDAQ", "EV",       "USA", 200, 5, "multi_site",       3,  True,  "Marcum LLP",         "mid",     6.0,  True),
    ("BNGO",  "Bionano Genomics",           "NASDAQ", "Biotech",  "USA", 180, 1, "real_office",      10, True,  "Ernst & Young",      "big4",    10.0, True),
]

COLUMNS = [
    "ticker", "name", "exchange", "sector", "country_iso3",
    "employee_count", "location_count", "location_type",
    "named_customers_count", "has_shipped_product",
    "auditor", "auditor_tier", "years_operating", "ceo_verifiable",
]


def main():
    out = Path(__file__).parent / "company_seeds.parquet"
    df = pl.DataFrame(
        {col: [row[i] for row in SEEDS] for i, col in enumerate(COLUMNS)}
    )
    df.write_parquet(out)
    print(f"Written {len(df)} seeds → {out}")
    print(df.select(["ticker", "sector", "auditor_tier"]))


if __name__ == "__main__":
    main()
