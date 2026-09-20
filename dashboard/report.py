"""
The report definition: layout, visuals, and per-visual formatting.

Power BI stores a visual's formatting as a JSON *string* inside the report
JSON, and every property value is wrapped in an expression literal. The helpers
at the top of this file hide that so the layout below reads as layout.

Unknown formatting properties are ignored by Power BI rather than rejected, so
the styling here fails soft: if a property name is wrong for a given build the
visual still renders, just without that one refinement.

Layout is a strict grid - 24px page margin, 16px gutters, 1280x720 canvas -
because alignment is most of what separates a designed dashboard from a
placed one.
"""

import json
import uuid

from theme import (
    AXIS, BLUE, CANVAS, CARD, FONT, FONT_LIGHT, FONT_SEMIBOLD, GRID,
    HAIRLINE, INK, INK_2, MUTED, ORANGE, SURFACE,
)

S = "scored_transactions"
D = "dashboard_summary"

PAGE_W, PAGE_H = 1280, 720
MARGIN, GUTTER = 24, 16
CONTENT_W = PAGE_W - 2 * MARGIN          # 1232


# --- expression helpers ------------------------------------------------------

def lit(value):
    return {"expr": {"Literal": {"Value": value}}}


def text(value):
    return lit(f"'{value}'")


def num(value):
    return lit(f"{value}D")


def flag(value: bool):
    return lit("true" if value else "false")


def colour(hex_value):
    return {"solid": {"color": lit(f"'{hex_value}'")}}


def props(**kwargs):
    return [{"properties": dict(kwargs)}]


# --- query helpers -----------------------------------------------------------

def col(alias, entity, column):
    return {
        "Column": {"Expression": {"SourceRef": {"Source": alias}},
                   "Property": column},
        "Name": f"{entity}.{column}",
    }


def measure(alias, entity, name):
    return {
        "Measure": {"Expression": {"SourceRef": {"Source": alias}},
                    "Property": name},
        "Name": f"{entity}.{name}",
    }


def total(alias, entity, column):
    return {
        "Aggregation": {
            "Expression": {"Column": {
                "Expression": {"SourceRef": {"Source": alias}},
                "Property": column}},
            "Function": 0,
        },
        "Name": f"Sum({entity}.{column})",
    }


def ref(select):
    """The queryRef a projection uses to point at a Select entry."""
    return {"queryRef": select["Name"]}


def equals_filter(entity, column, value):
    """Visual-level 'column = value' filter, as a JSON string."""
    return json.dumps([{
        "name": uuid.uuid4().hex[:12],
        "expression": {"Column": {
            "Expression": {"SourceRef": {"Entity": entity}},
            "Property": column}},
        "filter": {
            "Version": 2,
            "From": [{"Name": "f", "Entity": entity, "Type": 0}],
            "Where": [{"Condition": {"Comparison": {
                "ComparisonKind": 0,
                "Left": {"Column": {
                    "Expression": {"SourceRef": {"Source": "f"}},
                    "Property": column}},
                "Right": {"Literal": {"Value": f"{value}L"}},
            }}}],
        },
        "type": "Categorical",
    }])


# --- visual construction -----------------------------------------------------

def _check_query(single):
    """
    Every Select must reference a source alias the From clause declares.

    A mismatch produces a query Power BI cannot bind, and the visual does not
    render - with no error on the visual itself, just a page-level "something
    went wrong". Cheap to assert here, near-invisible to debug in Desktop.
    """
    query = single.get("prototypeQuery")
    if not query:
        return
    declared = {f["Name"] for f in query["From"]}
    for select in query["Select"]:
        body = next(v for k, v in select.items() if k != "Name")
        # Aggregations wrap another expression one level down.
        if "Expression" in body and "Column" in body.get("Expression", {}):
            body = body["Expression"]["Column"]
        used = body["Expression"]["SourceRef"]["Source"]
        if used not in declared:
            raise ValueError(
                f"{single['visualType']}: Select {select['Name']!r} references "
                f"source {used!r}, but From declares {sorted(declared)}")

    names = {s["Name"] for s in query["Select"]}
    for role, items in single.get("projections", {}).items():
        for item in items:
            if item["queryRef"] not in names:
                raise ValueError(
                    f"{single['visualType']}: projection {role} references "
                    f"{item['queryRef']!r}, not in {sorted(names)}")

    for clause in query.get("OrderBy", []):
        used = clause["Expression"]["Column"]["Expression"]["SourceRef"]["Source"]
        if used not in declared:
            raise ValueError(
                f"{single['visualType']}: OrderBy references source {used!r}, "
                f"but From declares {sorted(declared)}")


def container(x, y, w, h, z, single, filters="[]"):
    _check_query(single)
    config = {
        "name": uuid.uuid4().hex[:20],
        "layouts": [{"id": 0, "position": {
            "x": x, "y": y, "z": z, "width": w, "height": h}}],
        "singleVisual": single,
    }
    return {"x": x, "y": y, "z": z, "width": w, "height": h,
            "config": json.dumps(config), "filters": filters}


def chart(x, y, w, h, z, visual_type, entity, selects, projections,
          title=None, objects=None, order_by=None, filters="[]",
          alias="q"):
    query = {
        "Version": 2,
        "From": [{"Name": alias, "Entity": entity, "Type": 0}],
        "Select": selects,
    }
    if order_by:
        query["OrderBy"] = order_by

    single = {
        "visualType": visual_type,
        "projections": projections,
        "prototypeQuery": query,
        "drillFilterOtherVisuals": True,
        "objects": objects or {},
    }
    if title is not None:
        single["vcObjects"] = {"title": props(
            show=flag(True), text=text(title),
            fontColor=colour(INK), fontSize=num(11),
            fontFamily=text(FONT_SEMIBOLD), alignment=text("left"))}
    return container(x, y, w, h, z, single, filters)


def textbox(x, y, w, h, z, runs, align="left", fill=None):
    """A static text block. `runs` is a list of (value, style-dict)."""
    single = {
        "visualType": "textbox",
        "drillFilterOtherVisuals": True,
        "objects": {"general": [{"properties": {"paragraphs": [{
            "horizontalTextAlignment": align,
            "textRuns": [{"value": v, "textStyle": st} for v, st in runs],
        }]}}]},
        "vcObjects": {
            "background": props(
                show=flag(fill is not None),
                color=colour(fill or CARD),
                transparency=num(0)),
            "border": props(show=flag(False)),
            "title": props(show=flag(False)),
            "visualHeader": props(show=flag(False)),
        },
    }
    return container(x, y, w, h, z, single)


def style(size, weight="normal", color=INK, family=FONT):
    return {"fontFamily": family, "fontSize": f"{size}pt",
            "fontWeight": weight, "color": color}


def header(title, subtitle, z=0):
    """
    Full-width dark band, with the text laid on top rather than inside it.

    A textbox insets its own text, so putting the title in the band itself
    would leave it flush near the canvas edge while everything below sits on
    the 24px margin. Separating the band from the type keeps one vertical
    line down the left of the page.
    """
    return [
        textbox(0, 0, PAGE_W, 76, z, [(" ", style(1, "normal", INK))], fill=INK),
        textbox(MARGIN - 12, 14, PAGE_W - 2 * MARGIN, 30, z + 1,
                [(title, style(16, "600", "#FFFFFF", FONT_SEMIBOLD))]),
        textbox(MARGIN - 12, 44, PAGE_W - 2 * MARGIN, 24, z + 2,
                [(subtitle, style(9, "normal", "#A9A8A3"))]),
    ]


def kpi(x, y, w, h, z, measure_name, title):
    # The alias here MUST match the one `chart` puts in the query's From clause.
    # A Select that references source "s" while From declares "q" is an invalid
    # query, and the visual silently fails to render at all.
    selects = [measure("s", S, measure_name)]
    return chart(
        x, y, w, h, z, "card", S, selects,
        {"Values": [ref(selects[0])]},
        title=title,
        objects={
            "labels": props(color=colour(INK), fontSize=num(26),
                            fontFamily=text(FONT_LIGHT)),
            "categoryLabels": props(show=flag(False)),
            "wordWrap": props(show=flag(False)),
        },
        alias="s")


# --- page 1: operations ------------------------------------------------------

def page_operations():
    z = 0
    visuals = header(
        "Fraud review queue — PaySim holdout",
        "818,514 transactions · 4,570 frauds · scored by the leak-free Track D model "
        "(holdout PR-AUC 0.3260)")
    z = 3

    # KPI row. Four equal cards across the content width.
    card_w = (CONTENT_W - 3 * GUTTER) // 4        # 296
    kpis = [
        ("Fraud Exposure",  "Total fraud exposure"),
        ("Value Recovered", "Recovered by the queue"),
        ("Recovery Rate",   "Share of fraud value recovered"),
        ("Queue Precision", "Queue precision"),
    ]
    y = 100
    for i, (m, t) in enumerate(kpis):
        visuals.append(
            kpi(MARGIN + i * (card_w + GUTTER), y, card_w, 104, z, m, t))
        z += 1

    # Hero: the recovery curve. Log X, because budget spans 100 -> 50,000 and a
    # linear axis compresses everything interesting into the left edge.
    row_y = y + 104 + GUTTER                       # 220
    row_h = 248
    curve_w = 792
    q_size = col("q", D, "queue_size")
    q_val = total("q", D, "value_recovered_pct")
    q_lift = total("q", D, "lift_vs_random")
    visuals.append(chart(
        MARGIN, row_y, curve_w, row_h, z, "lineChart", D,
        [q_size, q_val, q_lift],
        {"Category": [ref(q_size)], "Y": [ref(q_val)],
         "Tooltips": [ref(q_lift)]},
        title="What the review queue recovers",
        objects={
            "categoryAxis": props(
                show=flag(True), axisScale=text("Log"),
                showAxisTitle=flag(True), titleText=text("Review budget (transactions)"),
                labelColor=colour(INK_2), fontSize=num(9),
                gridlineShow=flag(False)),
            "valueAxis": props(
                show=flag(True), showAxisTitle=flag(True),
                titleText=text("Fraud value recovered (%)"),
                start=num(0), end=num(100),
                labelColor=colour(INK_2), fontSize=num(9),
                gridlineColor=colour(GRID), gridlineThickness=num(1)),
            "dataPoint": props(fill=colour(BLUE)),
            "lineStyles": props(
                strokeWidth=num(3), showMarker=flag(True),
                markerSize=num(5), lineStyle=text("solid")),
            "labels": props(show=flag(False)),
            "legend": props(show=flag(False)),
        }))
    z += 1

    # Donut: of all fraud, what did the queue catch. Filtered to is_fraud = 1,
    # or the 813k correctly-ignored rows swamp it.
    donut_x = MARGIN + curve_w + GUTTER
    donut_w = CONTENT_W - curve_w - GUTTER         # 424
    d_out = col("s", S, "outcome")
    d_txn = measure("s", S, "Transactions")
    visuals.append(chart(
        donut_x, row_y, donut_w, row_h, z, "donutChart", S,
        [d_out, d_txn],
        {"Category": [ref(d_out)], "Y": [ref(d_txn)]},
        title="Of all fraud, what the queue caught",
        filters=equals_filter(S, "is_fraud", 1),
        objects={
            "legend": props(show=flag(True), position=text("Bottom"),
                            showTitle=flag(False), fontSize=num(9),
                            labelColor=colour(INK_2)),
            "labels": props(show=flag(True), fontSize=num(9),
                            color=colour(INK_2),
                            labelStyle=text("Percent of total")),
            "slices": props(innerRadiusRatio=num(62)),
        },
        alias="s"))
    z += 1

    # The queue itself, in rank order - what an analyst actually works through.
    table_y = row_y + row_h + GUTTER               # 484
    table_h = PAGE_H - table_y - MARGIN            # 212
    cols = [col("s", S, c) for c in
            ("score_rank", "fraud_score", "type", "amount", "nameDest",
             "dest_txn_count", "outcome")]
    visuals.append(chart(
        MARGIN, table_y, CONTENT_W, table_h, z, "tableEx", S,
        cols, {"Values": [ref(c) for c in cols]},
        title="The review queue — top 1,000 by score, ties broken by amount",
        filters=equals_filter(S, "in_review_queue", 1),
        order_by=[{"Direction": 1, "Expression": {"Column": {
            "Expression": {"SourceRef": {"Source": "s"}},
            "Property": "score_rank"}}}],
        objects={
            "grid": props(
                gridVertical=flag(False), gridHorizontal=flag(True),
                gridHorizontalColor=colour(GRID),
                outlineColor=colour(HAIRLINE), rowPadding=num(3)),
            "columnHeaders": props(
                fontColor=colour(INK_2), backColor=colour(SURFACE),
                fontSize=num(9), bold=flag(True), outline=text("BottomOnly")),
            "values": props(
                fontColor=colour(INK), fontSize=num(9),
                backColorPrimary=colour(CARD),
                backColorSecondary=colour(SURFACE)),
        },
        alias="s"))

    return visuals


# --- page 2: model behaviour -------------------------------------------------

def page_behaviour():
    visuals = header(
        "Model behaviour — and the artifact that is not signal",
        "Track D is the leak-free feature set. The hourly pattern below is the "
        "simulator's, not a fraud pattern — it is documented here, not modelled.")
    z = 3

    # Slicers, in one row, so filters read as a control strip rather than
    # competing with the charts.
    slicer_w = 232
    y = 100
    for i, (column, label) in enumerate(
            [("type", "Transaction type"), ("risk_band", "Risk band")]):
        c = col("s", S, column)
        visuals.append(chart(
            MARGIN + i * (slicer_w + GUTTER), y, slicer_w, 92, z,
            "slicer", S, [c], {"Values": [ref(c)]},
            title=label,
            objects={
                "general": props(outlineWeight=num(0),
                                 orientation=text("Horizontal")),
                "items": props(fontColor=colour(INK_2), fontSize=num(9),
                               background=colour(SURFACE)),
                "header": props(show=flag(False)),
            },
            alias="s"))
        z += 1

    # Capacity table, beside the slicers - the numeric backing for the curve.
    cap_x = MARGIN + 2 * (slicer_w + GUTTER)
    cap_cols = [col("q", D, c) for c in
                ("queue_size", "frauds_caught", "value_recovered_pct",
                 "precision", "lift_vs_random")]
    visuals.append(chart(
        cap_x, y, PAGE_W - MARGIN - cap_x, 92 + GUTTER + 240, z, "tableEx", D,
        cap_cols, {"Values": [ref(c) for c in cap_cols]},
        title="Review capacity — what each budget returns",
        order_by=[{"Direction": 1, "Expression": {"Column": {
            "Expression": {"SourceRef": {"Source": "q"}},
            "Property": "queue_size"}}}],
        objects={
            "grid": props(gridVertical=flag(False), gridHorizontal=flag(True),
                          gridHorizontalColor=colour(GRID),
                          outlineColor=colour(HAIRLINE), rowPadding=num(6)),
            "columnHeaders": props(fontColor=colour(INK_2),
                                   backColor=colour(SURFACE),
                                   fontSize=num(9), bold=flag(True),
                                   outline=text("BottomOnly")),
            "values": props(fontColor=colour(INK), fontSize=num(9),
                            backColorPrimary=colour(CARD),
                            backColorSecondary=colour(SURFACE)),
        }))
    z += 1

    # Fraud rate by hour - the timing artifact, in orange because it is the
    # thing being warned about rather than a result.
    chart_y = y + 92 + GUTTER                      # 208
    chart_w = 2 * slicer_w + GUTTER                # 480
    h_hour = col("s", S, "hour_of_day")
    m_rate = measure("s", S, "Fraud Rate")
    visuals.append(chart(
        MARGIN, chart_y, chart_w, 240, z, "columnChart", S,
        [h_hour, m_rate],
        {"Category": [ref(h_hour)], "Y": [ref(m_rate)]},
        title="Fraud rate by hour — the simulator's artifact, not a pattern",
        objects={
            "dataPoint": props(fill=colour(ORANGE)),
            "categoryAxis": props(show=flag(True), labelColor=colour(INK_2),
                                  fontSize=num(8), gridlineShow=flag(False),
                                  showAxisTitle=flag(False)),
            "valueAxis": props(show=flag(True), labelColor=colour(INK_2),
                               fontSize=num(8), gridlineColor=colour(GRID),
                               showAxisTitle=flag(False)),
            "labels": props(show=flag(False)),
            "legend": props(show=flag(False)),
        },
        alias="s"))
    z += 1

    # Outcome mix by risk band - does the banding actually separate anything.
    lower_y = chart_y + 240 + GUTTER               # 464
    lower_h = PAGE_H - lower_y - MARGIN            # 232
    b_band = col("s", S, "risk_band")
    b_out = col("s", S, "outcome")
    b_txn = measure("s", S, "Transactions")
    visuals.append(chart(
        MARGIN, lower_y, CONTENT_W, lower_h, z, "columnChart", S,
        [b_band, b_out, b_txn],
        {"Category": [ref(b_band)], "Series": [ref(b_out)],
         "Y": [ref(b_txn)]},
        title="Outcomes by risk band — fraud only",
        filters=equals_filter(S, "is_fraud", 1),
        objects={
            "categoryAxis": props(show=flag(True), labelColor=colour(INK_2),
                                  fontSize=num(9), gridlineShow=flag(False),
                                  showAxisTitle=flag(False)),
            "valueAxis": props(show=flag(True), labelColor=colour(INK_2),
                               fontSize=num(9), gridlineColor=colour(GRID),
                               showAxisTitle=flag(False)),
            "legend": props(show=flag(True), position=text("Top"),
                            showTitle=flag(False), fontSize=num(9),
                            labelColor=colour(INK_2)),
            "labels": props(show=flag(True), fontSize=num(8),
                            color=colour(INK_2)),
        },
        alias="s"))

    return visuals


# --- assembly ----------------------------------------------------------------

def section(ordinal, name, display, visuals):
    return {
        "id": ordinal,
        "name": name,
        "displayName": display,
        "filters": "[]",
        "ordinal": ordinal,
        "visualContainers": visuals,
        "config": json.dumps({"objects": {
            "background": props(color=colour(CANVAS), transparency=num(0)),
            "outspace": props(color=colour(CANVAS)),
        }}),
        "displayOption": 1,
        "width": PAGE_W,
        "height": PAGE_H,
    }


def build(theme_name: str) -> dict:
    return {
        "id": 0,
        "resourcePackages": [{
            "resourcePackage": {
                "disabled": False,
                "items": [{"name": theme_name,
                           "path": f"{theme_name}.json",
                           "type": 202}],
                "name": "RegisteredResources",
                "type": 1,
            }
        }],
        "sections": [
            section(0, "ReportSectionOps", "Operations", page_operations()),
            section(1, "ReportSectionModel", "Model behaviour", page_behaviour()),
        ],
        "config": json.dumps({
            "version": "5.43",
            "themeCollection": {
                "baseTheme": {"name": "CY24SU10", "version": "5.55", "type": 2},
                "customTheme": {"name": theme_name, "version": "5.55", "type": 1},
            },
            "activeSectionIndex": 0,
            "defaultDrillFilterOtherVisuals": True,
            "settings": {
                "useStylableVisualContainerHeader": True,
                "hideVisualContainerHeader": True,
            },
        }),
        "layoutOptimization": 0,
        "publicCustomVisuals": [],
    }
