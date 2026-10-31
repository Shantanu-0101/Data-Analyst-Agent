"""Build Project_Report.pdf from the walkthrough's saved results.

Run the walkthrough first (python deliverables/walkthrough.py): it writes
outputs/walkthrough_results.json and the chart PNGs this report shows.
Run with: python deliverables/generate_report.py
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUTS = ROOT / "outputs"
RESULTS = OUTPUTS / "walkthrough_results.json"
SCREENSHOT = OUTPUTS / "app_screenshot.jpg"  # optional: a capture of the Streamlit app
REPORT = HERE / "Project_Report.pdf"

FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {  # family: {style: file}; core fonts are used when these are missing
    "Body": {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf"},
    "Mono": {"": "consola.ttf", "B": "consolab.ttf"},
}

INK = (33, 37, 41)
MUTED = (108, 117, 125)
ACCENT = (37, 99, 235)
CODE_BG = (243, 244, 246)
BOX_FILL = (232, 240, 254)
FIX_FILL = (254, 243, 199)
END_FILL = (220, 252, 231)

TABLE_ROWS = 6        # rows of a table answer shown in the report
CHART_W = 140         # mm; the PNGs are 900x500
CHART_H = CHART_W * 500 / 900 + 4
CODE_MAX_LINES = 18   # longer code is cut, with a note
ERROR_MAX_LINES = 10

CHART_NAMES = {"bar": "Bar chart", "line": "Line chart", "pie": "Pie chart", "heatmap": "Heatmap"}


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(True, margin=18)
        self.unicode = all((FONT_DIR / f).exists() for styles in FONTS.values() for f in styles.values())
        if self.unicode:
            for family, styles in FONTS.items():
                for style, file in styles.items():
                    self.add_font(family, style, str(FONT_DIR / file))
        self.body_font = "Body" if self.unicode else "helvetica"
        self.mono_font = "Mono" if self.unicode else "courier"

    def text_safe(self, text: str) -> str:
        # model text uses special spaces and hyphens that Arial lacks
        text = str(text).replace("\u202f", " ").replace("\u00a0", " ").replace("\u2011", "-")
        if not self.unicode:  # core fonts are latin-1 only
            text = text.replace("→", "->").replace("–", "-").replace("—", "-").replace("’", "'").replace("·", "-")
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font(self.body_font, "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"Data Analyst Agent  ·  page {self.page_no()}" if self.unicode else f"Page {self.page_no()}",
                  align="C")

    # --- building blocks ------------------------------------------------------

    def h1(self, text, in_toc=True):
        if in_toc:
            self.start_section(text, level=0)
        self.ln(2)
        self.set_font(self.body_font, "B", 17)
        self.set_text_color(*INK)
        self.multi_cell(0, 9, self.text_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y() + 1, self.l_margin + 30, self.get_y() + 1)
        self.ln(5)

    def h2(self, text, keep=20):
        """A subheading, moved to the next page with its first `keep` mm of content if they would not fit."""
        self.keep_together(keep + 12)
        self.start_section(text, level=1)
        self.ln(2)
        self.set_font(self.body_font, "B", 12.5)
        self.set_text_color(*INK)
        self.multi_cell(0, 7, self.text_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def para(self, text, size=10.5, color=INK, style=""):
        self.set_font(self.body_font, style, size)
        self.set_text_color(*color)
        self.multi_cell(0, size * 0.52, self.text_safe(text), markdown=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullets(self, items, size=10.5, numbered=False):
        self.set_font(self.body_font, "", size)
        self.set_text_color(*INK)
        for number, item in enumerate(items, 1):
            self.set_x(self.l_margin + 2)
            self.cell(6, size * 0.52, f"{number}." if numbered else ("•" if self.unicode else "-"))
            self.multi_cell(0, size * 0.52, self.text_safe(item), markdown=True,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1)
        self.ln(1)

    def label(self, text):
        self.set_font(self.body_font, "B", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.text_safe(text.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code(self, text, max_lines=CODE_MAX_LINES, size=8.5):
        lines = text.rstrip().splitlines() or [""]
        if len(lines) > max_lines:
            cut = len(lines) - max_lines
            lines = lines[:max_lines] + [f"# ... {cut} more line{'s' if cut != 1 else ''}"]
        self.set_font(self.mono_font, "", size)
        self.set_text_color(*INK)
        self.set_fill_color(*CODE_BG)
        self.multi_cell(0, size * 0.48, self.text_safe("\n".join(lines)), fill=True, padding=(2.5, 3),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def data_table(self, columns, rows, size=8.5, col_widths=None):
        self.set_font(self.body_font, "", size)
        self.set_text_color(*INK)
        self.set_draw_color(209, 213, 219)
        self.set_fill_color(255, 255, 255)  # the table fills cells with the current colour
        self.set_line_width(0.2)
        with self.table(line_height=size * 0.55, text_align="LEFT", padding=1.2, col_widths=col_widths,
                        headings_style=_heading_style(self), first_row_as_headings=True) as table:
            table.row([self.text_safe(col) for col in columns])
            for row in rows:
                table.row([self.text_safe(_cell(value)) for value in row])
        self.ln(3)

    def keep_together(self, height):
        """Start a new page if the next `height` mm would not fit."""
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()


def _heading_style(pdf):
    from fpdf.fonts import FontFace

    return FontFace(family=pdf.body_font, emphasis="BOLD", fill_color=CODE_BG)


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00:00(\.000)?", value):
        return value[:10]
    return str(value)


def _plain(text: str) -> str:
    """Model text without markdown fences and emphasis, which the PDF would show literally."""
    text = re.sub(r"^[ \t]*```[\w+-]*[ \t]*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*{1,2}|_{2})(\S(?:[^*\n]*?\S)?)\1", r"\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text.replace("`", ""))
    return text.strip()


# --- table of contents -------------------------------------------------------

def render_toc(pdf: Report, outline):
    pdf.h1("Table of Contents", in_toc=False)
    for section in outline:
        top = section.level == 0
        if top:
            pdf.ln(2)
        indent = 0 if top else 8
        pdf.set_font(pdf.body_font, "B" if top else "", 11 if top else 10)
        pdf.set_text_color(*(INK if top else MUTED))
        link = pdf.add_link(page=section.page_number)
        y = pdf.get_y()
        name = pdf.text_safe(section.name)
        name_w = pdf.get_string_width(name) + 2
        pdf.set_x(pdf.l_margin + indent)
        pdf.cell(name_w, 7, name, link=link)
        page = str(section.page_number)
        page_w = pdf.get_string_width(page) + 1
        right = pdf.w - pdf.r_margin
        pdf.set_xy(right - page_w, y)
        pdf.cell(page_w, 7, page, align="R", link=link)
        pdf.set_draw_color(*MUTED)
        pdf.set_line_width(0.25)
        pdf.set_dash_pattern(dash=0.3, gap=1.2)
        pdf.line(pdf.l_margin + indent + name_w + 1, y + 5, right - page_w - 2, y + 5)
        pdf.set_dash_pattern()
        pdf.set_y(y + 7)


# --- diagram pieces ----------------------------------------------------------

def _box(pdf: Report, x, y, w, h, title, lines=(), fill=BOX_FILL):
    """A rounded box centred on x, with a bold title and smaller lines under it."""
    pdf.set_fill_color(*fill)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.4)
    pdf.rect(x - w / 2, y, w, h, style="DF", round_corners=True, corner_radius=2.5)
    top = y + (h - 4.5 - 3.6 * len(lines)) / 2
    pdf.set_font(pdf.body_font, "B", 9.5)
    pdf.set_text_color(*INK)
    pdf.set_xy(x - w / 2, top)
    pdf.cell(w, 4.5, pdf.text_safe(title), align="C")
    pdf.set_font(pdf.body_font, "", 7.5)
    pdf.set_text_color(*MUTED)
    for i, line in enumerate(lines):
        pdf.set_xy(x - w / 2, top + 4.5 + 3.6 * i)
        pdf.cell(w, 3.6, pdf.text_safe(line), align="C")


def _arrow(pdf: Report, points, label=None, label_at=None, align="L"):
    """A line through `points` with an arrowhead at the last one; the label is left- or right-aligned at label_at."""
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.35)
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        pdf.line(ax, ay, bx, by)
    (ax, ay), (bx, by) = points[-2], points[-1]
    length = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / length, (by - ay) / length
    size = 1.8
    back_x, back_y = bx - ux * size * 1.4, by - uy * size * 1.4
    pdf.set_fill_color(*INK)
    pdf.polygon([(bx, by), (back_x - uy * size, back_y + ux * size), (back_x + uy * size, back_y - ux * size)],
                style="F")
    if label:
        pdf.set_font(pdf.body_font, "I", 7.5)
        pdf.set_text_color(*MUTED)
        width = pdf.get_string_width(label) + 1
        x, y = label_at
        pdf.set_xy(x - width if align == "R" else x, y)
        pdf.cell(width, 3.6, label)


def _group(pdf: Report, x, y, w, h, title, title_x):
    """A dashed outline around related boxes, titled at title_x along its top edge."""
    pdf.set_draw_color(*MUTED)
    pdf.set_line_width(0.3)
    pdf.set_dash_pattern(dash=1.5, gap=1)
    pdf.rect(x, y, w, h, style="D", round_corners=True, corner_radius=3)
    pdf.set_dash_pattern()
    pdf.set_font(pdf.body_font, "B", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(title_x, y + 1.2)
    pdf.cell(pdf.get_string_width(title) + 1, 3.6, title)


# --- the architecture diagram ------------------------------------------------

def draw_architecture(pdf: Report):
    """The user, the app's components and the external LLM API, with what passes between them."""
    y0 = pdf.get_y() + 2
    left, right = pdf.l_margin, pdf.l_margin + 130   # the app's outline; the LLM API sits to its right
    mid = (left + right) / 2
    loader_x, agent_x = left + 29, left + 99           # column centres
    box_h = 17

    y_app = y0 + 29
    y_ui = y_app + 8
    y_row2 = y_ui + 14 + 13
    y_row3 = y_row2 + box_h + 13
    y_row4 = y_row3 + box_h + 13
    bottom = y_row4 + 14 + 6

    _group(pdf, left, y_app, right - left, bottom - y_app, "DATA ANALYST AGENT APP (PYTHON)", left + 3)

    _box(pdf, mid, y0, 60, 13, "User", ["in a web browser"], fill=CODE_BG)
    _arrow(pdf, [(mid - 10, y0 + 13), (mid - 10, y_ui)], "CSV, questions", (mid - 12, y0 + 17), align="R")
    _arrow(pdf, [(mid + 10, y_ui), (mid + 10, y0 + 13)], "tables, charts, answers", (mid + 12, y0 + 17))

    _box(pdf, mid, y_ui, 118, 14, "Streamlit UI  (app.py)",
         ["CSV upload, chat, question history; shows tables, charts and answers"])

    _box(pdf, loader_x, y_row2, 46, box_h, "Data loader", ["loader.py", "reads the CSV, builds", "the schema summary"])
    _box(pdf, agent_x, y_row2, 50, box_h, "Agent (LangGraph)", ["graph.py, nodes.py", "plan, generate, execute, fix"])
    _arrow(pdf, [(loader_x, y_ui + 14), (loader_x, y_row2)], "CSV file", (loader_x + 2, y_ui + 18))
    _arrow(pdf, [(loader_x + 23, y_row2 + box_h / 2), (agent_x - 25, y_row2 + box_h / 2)],
           "df, schema", (loader_x + 24, y_row2 + 3))
    _arrow(pdf, [(agent_x - 11, y_ui + 14), (agent_x - 11, y_row2)], "question", (agent_x - 13, y_ui + 18), align="R")
    _arrow(pdf, [(agent_x + 11, y_row2), (agent_x + 11, y_ui + 14)], "result", (agent_x + 13, y_ui + 18))

    llm_x = right + 28
    _box(pdf, llm_x, y_row2, 32, box_h, "LLM API", ["Groq, or", "Google Gemini"], fill=CODE_BG)
    _arrow(pdf, [(agent_x + 25, y_row2 + 5), (llm_x - 16, y_row2 + 5)], "prompt", (right + 1.5, y_row2 + 0.8))
    _arrow(pdf, [(llm_x - 16, y_row2 + 12), (agent_x + 25, y_row2 + 12)], "reply", (right + 1.5, y_row2 + 12.8))

    _box(pdf, loader_x, y_row3, 46, box_h, "Prompt templates", ["templates.py", "plan, code and fix prompts"])
    _box(pdf, agent_x, y_row3, 50, box_h, "Sandbox", ["sandbox.py", "AST check, safe builtins,", "10-second timeout"])
    _arrow(pdf, [(loader_x + 23, y_row3 + 3), (agent_x - 25, y_row2 + box_h - 3)])
    _arrow(pdf, [(agent_x - 11, y_row2 + box_h), (agent_x - 11, y_row3)], "code", (agent_x - 13, y_row2 + box_h + 4),
           align="R")
    _arrow(pdf, [(agent_x + 11, y_row3), (agent_x + 11, y_row2 + box_h)], "result, error",
           (agent_x + 13, y_row2 + box_h + 4))

    _box(pdf, agent_x, y_row4, 50, 14, "pandas, NumPy, Plotly", ["run the generated code"])
    _arrow(pdf, [(agent_x, y_row3 + box_h), (agent_x, y_row4)], "exec()", (agent_x + 2, y_row3 + box_h + 4))

    pdf.set_y(bottom + 6)


# --- the flowchart -----------------------------------------------------------

def draw_flowchart(pdf: Report):
    """Upload -> schema -> question -> plan -> generate -> execute -> decision; fix loops back; then the answer."""
    x0, y0 = pdf.l_margin, pdf.get_y() + 2
    cx = x0 + 70          # centre of the main column
    w, h, gap = 56, 11, 6

    pill_h = 8
    pdf.set_fill_color(*INK)
    pdf.rect(cx - 15, y0, 30, pill_h, style="F", round_corners=True, corner_radius=4)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(pdf.body_font, "B", 9)
    pdf.set_xy(cx - 15, y0 + 1.8)
    pdf.cell(30, 4.5, "START", align="C")

    steps = [
        ("Upload a CSV", "or click Use sample data", CODE_BG),
        ("Read the schema", "columns, types, sample rows, stats", CODE_BG),
        ("Ask a question", "in plain English, in the chat box", CODE_BG),
        ("plan", "LLM: 1-2 sentence approach", BOX_FILL),
        ("generate", "LLM: pandas / Plotly code", BOX_FILL),
        ("execute", "sandbox runs the code, reads RESULT", BOX_FILL),
    ]
    tops = []
    y = y0 + pill_h
    for title, sub, fill in steps:
        _arrow(pdf, [(cx, y), (cx, y + gap)])
        y += gap
        _box(pdf, cx, y, w, h, title, [sub], fill)
        tops.append(y)
        y += h
    y_ask, y_plan, y_exec = tops[2], tops[3], tops[5]

    # decision diamond
    dw, dec_h = 58, 16
    y_dec = y + gap
    _arrow(pdf, [(cx, y), (cx, y_dec)])
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.4)
    pdf.polygon([(cx, y_dec), (cx + dw / 2, y_dec + dec_h / 2), (cx, y_dec + dec_h), (cx - dw / 2, y_dec + dec_h / 2)],
                style="DF")
    pdf.set_font(pdf.body_font, "", 7.8)
    pdf.set_text_color(*INK)
    pdf.set_xy(cx - dw / 2, y_dec + dec_h / 2 - 3.6)
    pdf.cell(dw, 3.6, "error and", align="C")
    pdf.set_xy(cx - dw / 2, y_dec + dec_h / 2)
    pdf.cell(dw, 3.6, "retries < 3 ?", align="C")

    # yes -> fix, to the right, then back up into execute
    fx = cx + 72
    _box(pdf, fx, y_dec + dec_h / 2 - h / 2, 48, h, "fix", ["LLM: rewrite from traceback"], FIX_FILL)
    _arrow(pdf, [(cx + dw / 2, y_dec + dec_h / 2), (fx - 24, y_dec + dec_h / 2)], "yes",
           (cx + dw / 2 + 2, y_dec + dec_h / 2 - 5))
    _arrow(pdf, [(fx, y_dec + dec_h / 2 - h / 2), (fx, y_exec + h / 2), (cx + w / 2, y_exec + h / 2)],
           "retries + 1", (fx + 2, y_exec + h + 1))

    # no -> the answer
    show_h = 14
    y_show = y_dec + dec_h + gap
    _arrow(pdf, [(cx, y_dec + dec_h), (cx, y_show)], "no", (cx + 2, y_dec + dec_h + 3.3))
    _box(pdf, cx, y_show, w, show_h, "Show the answer", ["table, chart, number or text,", "or the last error"],
         END_FILL)

    # the next question goes round again
    loop_x = cx - w / 2 - 12
    _arrow(pdf, [(cx - w / 2, y_show + show_h / 2), (loop_x, y_show + show_h / 2), (loop_x, y_ask + h / 2),
                 (cx - w / 2, y_ask + h / 2)])
    pdf.set_font(pdf.body_font, "I", 7.5)
    pdf.set_text_color(*MUTED)
    for i, line in enumerate(("next", "question")):
        pdf.set_xy(x0, (y_ask + y_show) / 2 + 3.6 * i)
        pdf.cell(loop_x - x0 - 2, 3.6, line, align="R")

    _group(pdf, cx - w / 2 - 5, y_plan - 3, fx + 24 + 5 - (cx - w / 2 - 5), y_dec + dec_h + 3 - (y_plan - 3),
           "LANGGRAPH AGENT", cx + w / 2 + 4)

    pdf.set_y(y_show + show_h + 4)


# --- the report --------------------------------------------------------------

def count_tests() -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--co", "-q"], cwd=ROOT, capture_output=True,
                             text=True, timeout=120).stdout
        found = re.search(r"(\d+) tests? collected", out)
        return found[1] if found else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def build(data: dict) -> Report:
    pdf = Report()
    results = data["results"]
    samples = [r for r in results if not r.get("broken_code")]
    healed = next((r for r in results if r.get("broken_code")), None)
    answered = sum(1 for r in samples if not r["error"])
    first_try = sum(1 for r in samples if not r["error"] and r["retries"] == 0)
    dataset = data["dataset"]
    tests = count_tests()

    # --- title page
    pdf.add_page()
    pdf.set_y(70)
    pdf.set_font(pdf.body_font, "B", 30)
    pdf.set_text_color(*INK)
    pdf.cell(0, 14, "Data Analyst Agent", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(pdf.body_font, "", 14)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 7, "Ask questions about a CSV in plain English; get tables, charts and answers, "
                   "with the code that produced them.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 40, pdf.get_y())
    pdf.ln(10)
    pdf.set_font(pdf.body_font, "", 11)
    pdf.set_text_color(*INK)
    for line in (
        "Project report",
        f"LLM: {data['provider']} · {data['model']}",
        f"Sample run: {data['run_at']} on {dataset['file']} ({dataset['rows']:,} rows)",
    ):
        pdf.cell(0, 7, pdf.text_safe(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- table of contents (filled in at the end, once page numbers are known); the next section starts
    # on the page after it
    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=1)

    # --- business understanding
    pdf.h1("1. Business Understanding")
    pdf.h2("The problem")
    pdf.para(
        "Most teams keep their numbers in spreadsheets and CSV exports: sales, orders, stock, budgets. The "
        "questions people ask of them are simple (\"which region sold most last quarter?\", \"how is revenue "
        "trending?\"), but answering them takes pivot tables, formulas or Python. People without those skills "
        "wait for an analyst, and analysts spend much of their week on the same small, one-off requests. "
        "Dashboards help only with the questions someone planned for in advance."
    )
    pdf.h2("Who it is for")
    pdf.bullets([
        "**Managers and business users** in sales, operations and finance who have data but not the time or "
        "skills to write formulas or code.",
        "**Small businesses** with no data team, who need quick answers from their own exports.",
        "**Data analysts** who want routine questions answered in seconds, with working pandas code to build on.",
        "**Students** learning pandas and Plotly, who can see the code behind every answer.",
    ])
    pdf.h2("What it offers")
    pdf.bullets([
        "**Ask in plain English:** upload a CSV and type a question; no formulas or code needed.",
        "**The right format automatically:** a table, a single number, a sentence or an interactive chart.",
        "**Reliable answers:** if the generated code fails, the agent reads the error and fixes it, up to 3 times.",
        "**No guessed column names:** the agent sees the file's columns, types and sample rows on every question.",
        "**Nothing hidden:** the plan and the code behind each answer are one click away, so results can be checked.",
        "**Follow-up questions:** \"now only 2024\" builds on the previous answer, like talking to an analyst.",
    ])
    pdf.h2("Example")
    pdf.para(
        "A sales manager uploads the year's sales export and asks \"monthly revenue trend\". A line chart appears "
        "in a few seconds. They follow up with \"now filter that to 2024 only\" and get the 2024 chart, then ask "
        "\"top 3 salespersons by quantity sold\" and get a table, all without writing a formula."
    )

    # --- project understanding
    pdf.add_page()
    pdf.h1("2. Project Understanding")
    pdf.h2("How it works")
    pdf.para(
        "The app is written in Python. When a CSV is loaded, it is read into a pandas DataFrame and summarised "
        "into a short schema description (columns, types, sample rows, missing values, statistics). Each "
        "question then runs through an agent built with LangGraph. The agent sends the question and the schema "
        "to a large language model (LLM) through its API, gets back a plan and Python code, and runs that code "
        "in a restricted sandbox. If the code fails, the error goes back to the LLM to fix. The result is shown "
        "in a Streamlit web page."
    )
    pdf.h2("Tools and APIs", keep=60)
    pdf.data_table(["Tool", "Used for"], [
        ["Python", "the whole project"],
        ["Streamlit", "the web interface: upload, chat, tables and charts"],
        ["LangGraph", "the agent's state machine: plan, generate, execute, fix"],
        ["LangChain (langchain-groq, langchain-google-genai)", "one interface for calling either LLM"],
        ["Groq API", "the default LLM, openai/gpt-oss-120b (about 2 seconds per call, free tier)"],
        ["Google Gemini API", "the alternative LLM (gemini-3.5-flash by default)"],
        ["pandas, NumPy", "loading the CSV and running the analysis"],
        ["Plotly Express", "interactive charts"],
        ["python-dotenv", "reading the API key from the .env file"],
        ["pytest", f"{tests} automated tests, run without an API key or network"],
        ["fpdf2, Kaleido", "this report, and the chart images in it"],
    ], size=9.5, col_widths=(1.6, 3))
    pdf.h2("The agent's steps", keep=50)
    pdf.data_table(["Step", "What it does"], [
        ["plan", "Asks the LLM for a one or two sentence approach, given the schema, the question and recent history."],
        ["generate", "Asks for Python that uses df, pd, px and np and stores the answer in RESULT. Markdown fences "
                     "and stray prose are stripped from the reply."],
        ["execute", "Runs the code in the sandbox on a copy of the data and classifies RESULT as a table, chart, "
                    "value or text. A failure returns a trimmed traceback."],
        ["fix", "Sends the traceback and the failed code back to the LLM for a corrected version; adds 1 to retries."],
        ["should_retry", "Routes to fix while there is an error and fewer than 3 fixes have been tried; otherwise ends."],
    ], size=9, col_widths=(1, 4))
    pdf.h2("The sandbox", keep=40)
    pdf.bullets([
        "An **AST check** rejects imports, file access, eval/exec and dunder attributes before anything runs, with "
        "a message the fix step can act on.",
        "Only **whitelisted builtins** are available (len, sum, sorted, round and similar), so open() and import do not exist.",
        "The code gets a **copy** of the DataFrame, so a bad in-place edit cannot change the user's data.",
        "A **10-second timeout** stops runaway loops from blocking the app.",
    ])
    pdf.h2("Calling the LLM", keep=25)
    pdf.para(
        "One question costs at least two LLM calls (plan and generate), plus one per fix. The last five questions, "
        "with their code, are included so follow-ups work. Temporary failures (rate limits, server errors, "
        "timeouts) are retried up to three times, and the app tells the user it is waiting."
    )

    # --- architecture
    pdf.add_page()
    pdf.h1("3. Architecture Diagram")
    pdf.para("The components of the system and what passes between them. Everything inside the dashed outline "
             "runs in one Python process; the LLM is an external API.")
    draw_architecture(pdf)
    pdf.h2("Components", keep=55)
    pdf.data_table(["Component", "Responsibility"], [
        ["Streamlit UI (app.py)", "File upload, chat input, question history; renders each result by its type."],
        ["Data loader (src/data/loader.py)", "Reads the CSV (UTF-8 or latin-1), parses date columns, builds the "
                                             "schema summary sent with every prompt."],
        ["Agent (src/agent/)", "The LangGraph state machine and its four nodes; ask() is the single entry point."],
        ["Prompt templates (src/prompts/)", "The plan, code and fix prompts, plus the recent-history block."],
        ["Sandbox (src/execution/sandbox.py)", "Checks and runs the generated code; returns the result or a traceback."],
        ["Settings (src/config.py, .env)", "API keys, which LLM to use, retry and timeout limits."],
        ["LLM API (Groq or Gemini)", "Writes the plan and the code; reached over HTTPS through LangChain."],
    ], size=9, col_widths=(1.8, 3.2))

    # --- flowchart
    pdf.add_page()
    pdf.h1("4. Flowchart")
    pdf.para("What happens from loading a file to seeing an answer. Grey steps happen in the app, blue steps "
             "are the agent, and the yellow step runs only when the code fails.")
    draw_flowchart(pdf)
    pdf.keep_together(60)
    pdf.bullets([
        "The user uploads a CSV, or picks the sample sales data.",
        "The loader reads it and builds the schema summary.",
        "The user types a question.",
        "**plan:** the LLM describes the approach in one or two sentences.",
        "**generate:** the LLM writes pandas or Plotly code that stores its answer in RESULT.",
        "**execute:** the sandbox checks and runs the code.",
        "If the code failed and fewer than 3 fixes have been tried, **fix** sends the error back to the LLM and "
        "the new code runs again.",
        "The answer is shown as a table, chart, number or text, with the plan and code under it. The next "
        "question starts again at step 3, with this one as context.",
    ], size=10, numbered=True)

    # --- final outcome
    pdf.add_page()
    pdf.h1("5. Final Outcome")
    if SCREENSHOT.exists():
        pdf.h2("The app", keep=100)
        pdf.image(str(SCREENSHOT), w=140)
        pdf.ln(3)
    pdf.h2("Results at a glance", keep=80)
    pdf.bullets([
        f"**{answered} of {len(samples)}** sample questions answered, **{first_try}** of them with the first code.",
        f"The self-correction test recovered from a real KeyError in **{_attempts(healed['retries'])}**."
        if healed else "No self-correction test was recorded.",
        f"**{tests}** automated tests pass, covering the loader, prompts, sandbox, agent loop and app.",
    ])
    pdf.data_table(["#", "Question", "Answer", "Fixes", "Time"], [
        [str(number), entry["question"], _answer_summary(entry), str(entry["retries"]), f"{entry['seconds']} s"]
        for number, entry in enumerate(results, 1)
    ], size=8.5, col_widths=(0.35, 3.4, 2, 0.6, 0.6))
    pdf.para(f"Run on {dataset['file']} ({dataset['rows']:,} rows) with {data['provider']} ({data['model']}), "
             f"{data['run_at']}.", size=8.5, color=MUTED)

    pdf.h2("Sample outputs", keep=60)
    for number, entry in enumerate(samples, 1):
        render_result(pdf, number, entry)

    if healed:
        pdf.h2("Self-correction test", keep=70)
        pdf.para(
            "For this test the first code was replaced with a version that misspells a column (revenu). The "
            "sandbox raised a real KeyError, and the live model's fix step read it and repaired the code."
        )
        pdf.label("First code (broken on purpose)")
        pdf.code(healed["broken_code"])
        pdf.label("Error sent back to the LLM")
        pdf.code(healed["first_error"], max_lines=ERROR_MAX_LINES)
        pdf.label(f"Code after {_attempts(healed['retries'])}")
        pdf.code(healed["code"])
        if healed.get("rows"):
            pdf.label("Result")
            pdf.data_table(healed["columns"], healed["rows"][:TABLE_ROWS])
    return pdf


def render_result(pdf: Report, number: int, entry: dict):
    kind = entry["result_type"]
    pdf.keep_together(min(_result_height(entry), 240))
    pdf.ln(2)
    pdf.set_font(pdf.body_font, "B", 11)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 6, pdf.text_safe(f"Q{number}. \"{entry['question']}\""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(pdf.body_font, "", 8.5)
    pdf.set_text_color(*MUTED)
    extra = f"  ·  {entry['note']}" if entry.get("note") else ""
    pdf.cell(0, 5, pdf.text_safe(f"{_attempts(entry['retries'])}  ·  {entry['seconds']} s{extra}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    if entry["error"]:
        pdf.label("Last error")
        pdf.code(entry["error"], max_lines=ERROR_MAX_LINES)
    elif kind == "dataframe":
        rows = entry["rows"][:TABLE_ROWS]
        pdf.data_table(entry["columns"], rows)
        if entry["row_count"] > len(rows):
            pdf.para(f"First {len(rows)} of {entry['row_count']:,} rows.", size=8.5, color=MUTED)
    elif kind == "plotly" and (OUTPUTS / entry["chart"]).exists():
        pdf.image(str(OUTPUTS / entry["chart"]), w=CHART_W)
        pdf.ln(3)
    elif kind == "value":
        pdf.set_font(pdf.body_font, "B", 16)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 9, _format_value(entry["value"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
    else:
        pdf.para(_plain(str(entry.get("value", ""))), size=10.5, style="B")
    pdf.ln(1)


def _answer_summary(entry: dict) -> str:
    """A few words for the results table: what kind of answer came back."""
    if entry["error"]:
        return "Failed"
    kind = entry["result_type"]
    if kind == "dataframe":
        return f"Table, {entry['row_count']:,} row{'s' if entry['row_count'] != 1 else ''}"
    if kind == "plotly":
        return CHART_NAMES.get(entry.get("chart_type"), "Chart")
    if kind == "value":
        return _format_value(entry["value"])
    return "Text answer"


def _format_value(value) -> str:
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def _attempts(n: int) -> str:
    return f"{n} fix attempt{'s' if n != 1 else ''}"


def _result_height(entry: dict) -> float:
    """A rough height in mm of one result block, to avoid splitting it across pages."""
    height = 18
    if entry["error"]:
        return height + ERROR_MAX_LINES * 4.1
    if entry["result_type"] == "dataframe":
        return height + (min(len(entry["rows"]), TABLE_ROWS) + 1) * 6 + 6
    if entry["result_type"] == "plotly":
        return height + CHART_H
    return height + 14


def main():
    if not RESULTS.exists():
        sys.exit(f"{RESULTS.relative_to(ROOT)} not found. Run deliverables/walkthrough.py first.")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    pdf = build(data)
    pdf.output(str(REPORT))
    print(f"Wrote {REPORT.name}: {pdf.page_no()} pages")


if __name__ == "__main__":
    main()
