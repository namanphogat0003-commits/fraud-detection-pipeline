"""
The report theme.

Power BI's default look is not neutral - it has a palette, a font stack and a
chrome style of its own, and a report that accepts them looks like a Power BI
report rather than like this project. The figures in reports/figures/ already
have a defined visual language, set in src/evaluate.py, so the dashboard
inherits it exactly: same surface, same ink, same accent blue and orange, same
grid grey.

A theme is the highest-leverage piece of formatting available, because it sets
defaults for every visual at once - container fill, border radius, title
typography, and the categorical palette. Per-visual overrides in build_pbip.py
then handle only what genuinely differs.
"""

# Lifted verbatim from src/evaluate.py so the dashboard and the figures cannot
# drift apart.
SURFACE = "#FCFCFB"
INK = "#0B0B0B"
INK_2 = "#52514E"
MUTED = "#898781"
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
BLUE = "#2A78D6"
ORANGE = "#EB6834"

# Page canvas sits slightly warmer and darker than the cards, so white
# containers read as raised without needing heavy shadows.
CANVAS = "#F2F1EC"
CARD = "#FFFFFF"
HAIRLINE = "#E6E5DE"

FONT = "Segoe UI"
FONT_SEMIBOLD = "Segoe UI Semibold"
FONT_LIGHT = "Segoe UI Light"


def theme_json(name: str = "FraudAudit") -> dict:
    """A full Power BI theme document."""
    return {
        "name": name,

        # Blue first so the primary series is always the accent; orange second
        # so a two-series chart reads caught/missed without further styling.
        "dataColors": [BLUE, ORANGE, MUTED, AXIS, "#1F5FAA", "#D03B3B",
                       INK_2, GRID],

        "background": CARD,
        "foreground": INK,
        "foregroundNeutralSecondary": INK_2,
        "foregroundNeutralTertiary": MUTED,
        "backgroundLight": GRID,
        "backgroundNeutral": CANVAS,
        "tableAccent": BLUE,
        "good": BLUE,
        "neutral": MUTED,
        "bad": ORANGE,
        "maximum": BLUE,
        "minimum": GRID,

        "textClasses": {
            "title":    {"fontFace": FONT_SEMIBOLD, "fontSize": 12, "color": INK},
            "header":   {"fontFace": FONT_SEMIBOLD, "fontSize": 11, "color": INK},
            "label":    {"fontFace": FONT, "fontSize": 9, "color": INK_2},
            "callout":  {"fontFace": FONT_LIGHT, "fontSize": 30, "color": INK},
        },

        "visualStyles": {
            # Applies to every visual type and every card within it.
            "*": {
                "*": {
                    "background": [{
                        "show": True,
                        "color": {"solid": {"color": CARD}},
                        "transparency": 0,
                    }],
                    "border": [{
                        "show": True,
                        "color": {"solid": {"color": HAIRLINE}},
                        "radius": 10,
                    }],
                    "title": [{
                        "show": True,
                        "fontColor": {"solid": {"color": INK}},
                        "background": {"solid": {"color": CARD}},
                        "fontSize": 11,
                        "fontFamily": FONT_SEMIBOLD,
                        "alignment": "left",
                        "titleWrap": False,
                    }],
                    # The hover header adds focus/filter/more icons to every
                    # visual. Useful while authoring, noise on a finished page.
                    "visualHeader": [{"show": False}],
                    "spacing": [{"customizeSpacing": True, "spaceBelowTitle": 6}],
                    "padding": [{"top": 8, "bottom": 8, "left": 12, "right": 12}],
                    "legend": [{
                        "showTitle": False,
                        "fontSize": 9,
                        "labelColor": {"solid": {"color": INK_2}},
                        "fontFamily": FONT,
                    }],
                    "categoryAxis": [{
                        "showAxisTitle": False,
                        "fontSize": 9,
                        "labelColor": {"solid": {"color": INK_2}},
                        "fontFamily": FONT,
                        "gridlineShow": False,
                    }],
                    "valueAxis": [{
                        "showAxisTitle": False,
                        "fontSize": 9,
                        "labelColor": {"solid": {"color": INK_2}},
                        "fontFamily": FONT,
                        "gridlineColor": {"solid": {"color": GRID}},
                        "gridlineThickness": 1,
                    }],
                    "labels": [{
                        "fontSize": 9,
                        "color": {"solid": {"color": INK_2}},
                        "fontFamily": FONT,
                    }],
                }
            },

            "page": {
                "*": {
                    "background": [{
                        "color": {"solid": {"color": CANVAS}},
                        "transparency": 0,
                    }],
                    "outspace": [{"color": {"solid": {"color": CANVAS}}}],
                }
            },

            # KPI cards: the number is the content, so it gets the light weight
            # at large size and the label is suppressed - the visual title
            # already says what it is.
            "card": {
                "*": {
                    "labels": [{
                        "color": {"solid": {"color": INK}},
                        "fontSize": 28,
                        "fontFamily": FONT_LIGHT,
                    }],
                    "categoryLabels": [{"show": False}],
                }
            },

            "tableEx": {
                "*": {
                    "grid": [{
                        "gridVertical": False,
                        "gridHorizontal": True,
                        "gridHorizontalColor": {"solid": {"color": GRID}},
                        "outlineColor": {"solid": {"color": HAIRLINE}},
                        "rowPadding": 4,
                    }],
                    "columnHeaders": [{
                        "fontColor": {"solid": {"color": INK_2}},
                        "backColor": {"solid": {"color": SURFACE}},
                        "fontSize": 9,
                        "bold": True,
                        "fontFamily": FONT,
                        "outline": "BottomOnly",
                    }],
                    "values": [{
                        "fontColor": {"solid": {"color": INK}},
                        "fontSize": 9,
                        "fontFamily": FONT,
                        "backColorPrimary": {"solid": {"color": CARD}},
                        "backColorSecondary": {"solid": {"color": SURFACE}},
                        "urlIcon": False,
                    }],
                }
            },

            "slicer": {
                "*": {
                    "header": [{
                        "show": False,
                    }],
                    "items": [{
                        "fontColor": {"solid": {"color": INK_2}},
                        "fontSize": 9,
                        "fontFamily": FONT,
                    }],
                }
            },

            "textbox": {
                "*": {
                    "background": [{"show": False}],
                    "border": [{"show": False}],
                    "title": [{"show": False}],
                }
            },
        },
    }
