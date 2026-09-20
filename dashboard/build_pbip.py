"""
Generate the Power BI project that becomes dashboard/fraud_dashboard.pbix.

A .pbix cannot be written directly: its DataModel part is a compiled Analysis
Services tabular database in a proprietary compressed binary format, and only
the Power BI engine can produce one. PBIP is the supported text format that
Power BI Desktop reads and can save back out as .pbix, so that is what this
builds.

What you get:
  * a semantic model with both tables typed correctly and every measure from
    dashboard/README.md already defined
  * a two-page report matching the layout in that guide

Run with: python dashboard/build_pbip.py [--out DIR] [--project NAME]
Then: open dashboard/FraudDashboard.pbip in Power BI Desktop
      -> Refresh -> File -> Save As -> dashboard/fraud_dashboard.pbix
"""

import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as report_module          # noqa: E402
from theme import theme_json            # noqa: E402

THEME_NAME = "FraudAudit"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCORED_CSV = os.path.join(REPO, "data", "processed", "scored_transactions.csv")
SUMMARY_CSV = os.path.join(REPO, "reports", "dashboard_summary.csv")

# (name, TMDL dataType, format string or None, summarizeBy)
SCORED_COLUMNS = [
    ("step",            "int64",  "0",          "none"),
    ("hour_of_day",     "int64",  "0",          "none"),
    ("day",             "int64",  "0",          "none"),
    ("type",            "string", None,         "none"),
    ("amount",          "double", "#,0.00",     "sum"),
    ("nameDest",        "string", None,         "none"),
    ("oldbalanceDest",  "double", "#,0.00",     "sum"),
    ("dest_txn_count",  "int64",  "0",          "none"),
    ("is_fraud",        "int64",  "0",          "none"),
    ("fraud_score",     "double", "0.0000",     "none"),
    ("score_rank",      "int64",  "0",          "none"),
    ("in_review_queue", "int64",  "0",          "none"),
    ("risk_band",       "string", None,         "none"),
    ("outcome",         "string", None,         "none"),
]

SUMMARY_COLUMNS = [
    ("queue_size",          "int64",  "0",      "none"),
    ("pct_of_volume",       "double", "0.0000", "none"),
    ("frauds_caught",       "int64",  "0",      "sum"),
    ("frauds_total",        "int64",  "0",      "none"),
    ("value_recovered",     "double", "#,0",    "sum"),
    ("value_recovered_pct", "double", "0.0",    "none"),
    ("precision",           "double", "0.000",  "none"),
    ("lift_vs_random",      "double", "0.0",    "none"),
]

# The measures from dashboard/README.md, so they do not have to be retyped -
# which is where a transcription error would quietly change a reported number.
MEASURES = [
    ("Fraud Exposure",
     "CALCULATE(SUM(scored_transactions[amount]), scored_transactions[is_fraud] = 1)",
     "#,0,,\"M\""),
    ("Value Recovered",
     "CALCULATE(\n    SUM(scored_transactions[amount]),\n"
     "    scored_transactions[is_fraud] = 1,\n"
     "    scored_transactions[in_review_queue] = 1\n)",
     "#,0,,\"M\""),
    ("Recovery Rate",
     "DIVIDE([Value Recovered], [Fraud Exposure])",
     "0.0%"),
    ("Frauds Caught",
     "CALCULATE(\n    COUNTROWS(scored_transactions),\n"
     "    scored_transactions[is_fraud] = 1,\n"
     "    scored_transactions[in_review_queue] = 1\n)",
     "#,0"),
    ("Frauds Missed",
     "CALCULATE(\n    COUNTROWS(scored_transactions),\n"
     "    scored_transactions[is_fraud] = 1,\n"
     "    scored_transactions[in_review_queue] = 0\n)",
     "#,0"),
    ("Queue Size",
     "CALCULATE(COUNTROWS(scored_transactions), scored_transactions[in_review_queue] = 1)",
     "#,0"),
    ("Queue Precision",
     "DIVIDE([Frauds Caught], [Queue Size])",
     "0.0%"),
    ("Fraud Rate",
     "DIVIDE(\n    CALCULATE(COUNTROWS(scored_transactions), scored_transactions[is_fraud] = 1),\n"
     "    COUNTROWS(scored_transactions)\n)",
     "0.00%"),
    # Plain row count. The donut and the outcome breakdown need an unfiltered
    # count - Frauds Caught carries its own filter and would blank the
    # "missed" slice entirely.
    ("Transactions",
     "COUNTROWS(scored_transactions)",
     "#,0"),
]

TMDL_TYPE_TO_M = {"int64": "Int64.Type", "double": "type number", "string": "type text"}


def guid() -> str:
    return str(uuid.uuid4())


def m_query(csv_path: str, columns) -> str:
    """Power Query that loads one CSV with explicit types."""
    types = ", ".join(
        f'{{"{name}", {TMDL_TYPE_TO_M[dtype]}}}' for name, dtype, _, _ in columns)
    # M string literals do not escape backslashes, so a Windows path goes in as-is.
    return (
        "let\n"
        f'    Source = Csv.Document(File.Contents("{csv_path}"), '
        f"[Delimiter=\",\", Columns={len(columns)}, Encoding=65001, "
        "QuoteStyle=QuoteStyle.Csv]),\n"
        '    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n'
        f"    Typed = Table.TransformColumnTypes(Headers, {{{types}}})\n"
        "in\n"
        "    Typed"
    )


def table_tmdl(table_name: str, columns, csv_path: str, measures=()) -> str:
    """One table in TMDL. Indentation is tabs and is significant."""
    out = [f"table {table_name}", ""]

    for name, expression, fmt in measures:
        out.append(f"\tmeasure '{name}' = ```")
        for line in expression.split("\n"):
            out.append(f"\t\t\t{line}")
        out.append("\t\t\t```")
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out.append(f"\t\tlineageTag: {guid()}")
        out.append("")

    for name, dtype, fmt, summarize in columns:
        out.append(f"\tcolumn {name}")
        out.append(f"\t\tdataType: {dtype}")
        if fmt:
            out.append(f"\t\tformatString: {fmt}")
        out.append(f"\t\tlineageTag: {guid()}")
        out.append(f"\t\tsummarizeBy: {summarize}")
        out.append(f"\t\tsourceColumn: {name}")
        out.append("")
        out.append("\t\tannotation SummarizationSetBy = Automatic")
        out.append("")

    out.append(f"\tpartition {table_name} = m")
    out.append("\t\tmode: import")
    out.append("\t\tsource =")
    for line in m_query(csv_path, columns).split("\n"):
        out.append(f"\t\t\t\t{line}")
    out.append("")
    out.append("\tannotation PBI_ResultType = Table")
    out.append("")
    return "\n".join(out)


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def main(out_dir, project):
    model_dir = os.path.join(out_dir, f"{project}.SemanticModel")
    report_dir = os.path.join(out_dir, f"{project}.Report")
    written = []

    for csv in (SCORED_CSV, SUMMARY_CSV):
        if not os.path.exists(csv):
            raise FileNotFoundError(
                f"{csv} not found - run `python src/score_batch.py` first.")

    # --- .pbip entry point --------------------------------------------------
    written.append(write(os.path.join(out_dir, f"{project}.pbip"), json.dumps({
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{project}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    }, indent=2)))

    # --- semantic model -----------------------------------------------------
    written.append(write(os.path.join(model_dir, ".platform"), json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/"
                   "gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "SemanticModel", "displayName": project},
        "config": {"version": "2.0", "logicalId": guid()},
    }, indent=2)))

    written.append(write(os.path.join(model_dir, "definition.pbism"), json.dumps({
        "version": "4.2", "settings": {},
    }, indent=2)))

    written.append(write(
        os.path.join(model_dir, "definition", "database.tmdl"),
        "database\n\tcompatibilityLevel: 1567\n"))

    written.append(write(
        os.path.join(model_dir, "definition", "model.tmdl"),
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tsourceQueryCulture: en-GB\n"
        "\tdataAccessOptions\n"
        "\t\tlegacyRedirects\n"
        "\t\treturnErrorValuesAsNull\n"
        "\n"
        'annotation PBI_QueryOrder = ["scored_transactions","dashboard_summary"]\n'
        "\n"
        "ref table scored_transactions\n"
        "ref table dashboard_summary\n"))

    written.append(write(
        os.path.join(model_dir, "definition", "tables", "scored_transactions.tmdl"),
        table_tmdl("scored_transactions", SCORED_COLUMNS, SCORED_CSV, MEASURES)))

    written.append(write(
        os.path.join(model_dir, "definition", "tables", "dashboard_summary.tmdl"),
        table_tmdl("dashboard_summary", SUMMARY_COLUMNS, SUMMARY_CSV)))

    # --- report -------------------------------------------------------------
    written.append(write(os.path.join(report_dir, ".platform"), json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/"
                   "gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Report", "displayName": project},
        "config": {"version": "2.0", "logicalId": guid()},
    }, indent=2)))

    written.append(write(os.path.join(report_dir, "definition.pbir"), json.dumps({
        "version": "1.0",
        "datasetReference": {"byPath": {"path": f"../{project}.SemanticModel"}},
    }, indent=2)))

    written.append(write(
        os.path.join(report_dir, "StaticResources", "RegisteredResources",
                     f"{THEME_NAME}.json"),
        json.dumps(theme_json(THEME_NAME), indent=2)))

    written.append(write(os.path.join(report_dir, "report.json"),
                         json.dumps(report_module.build(THEME_NAME), indent=2)))

    print(f"Wrote {len(written)} files:")
    for path in written:
        print(f"  {os.path.relpath(path, REPO)}")
    print()
    print("Next:")
    print(f"  1. Open dashboard/{project}.pbip in Power BI Desktop")
    print("  2. Let it load both CSVs (Refresh if prompted)")
    print("  3. File -> Save As -> dashboard/fraud_dashboard.pbix")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    p.add_argument("--project", default="FraudDashboard")
    a = p.parse_args()
    main(a.out, a.project)
