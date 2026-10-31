"""Build Project_Report.pdf from the walkthrough's saved results.

Run the walkthrough first (walkthrough.py or build_notebook.py): it writes
outputs/walkthrough_results.json and the chart PNGs this report shows.
Run with: python deliverables/generate_report.py
"""
import json
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
ERROR_MAX_LINES = 12

QUESTION_KINDS = {"dataframe": "Table", "value": "Single value", "text": "Text answer", "plotly": "Chart"}


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

    def h1(self, text):
        self.ln(2)
        self.set_font(self.body_font, "B", 17)
        self.set_text_color(*INK)
        self.multi_cell(0, 9, self.text_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y() + 1, self.l_margin + 30, self.get_y() + 1)
        self.ln(5)

    def h2(self, text):
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

    def bullets(self, items, size=10.5):
        self.set_font(self.body_font, "", size)
        self.set_text_color(*INK)
        for item in items:
            self.set_x(self.l_margin + 2)
            self.cell(5, size * 0.52, "•" if self.unicode else "-")
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


# --- the flowchart -----------------------------------------------------------

def draw_flowchart(pdf: Report):
    """START -> plan -> generate -> execute -> decision; 'fix' loops back to execute; else END."""
    x0, y0 = pdf.l_margin, pdf.get_y() + 2
    cx = x0 + 62          # centre of the main column
    w, h, gap = 46, 11, 7

    def box(x, y, text, sub, fill, width=w):
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(0.4)
        pdf.rect(x - width / 2, y, width, h, style="DF", round_corners=True, corner_radius=2.5)
        pdf.set_text_color(*INK)
        pdf.set_font(pdf.body_font, "B", 10)
        pdf.set_xy(x - width / 2, y + 1.6)
        pdf.cell(width, 4.5, text, align="C")
        pdf.set_font(pdf.body_font, "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.set_xy(x - width / 2, y + 6)
        pdf.cell(width, 3.6, pdf.text_safe(sub), align="C")

    def arrow(points, text=None, text_at=None):
        pdf.set_draw_color(*INK)
        pdf.set_line_width(0.35)
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            pdf.line(ax, ay, bx, by)
        (ax, ay), (bx, by) = points[-2], points[-1]
        size = 1.8
        if ax == bx:   # vertical end
            d = 1 if by > ay else -1
            tri = [(bx, by), (bx - size, by - d * size * 1.4), (bx + size, by - d * size * 1.4)]
        else:          # horizontal end
            d = 1 if bx > ax else -1
            tri = [(bx, by), (bx - d * size * 1.4, by - size), (bx - d * size * 1.4, by + size)]
        pdf.set_fill_color(*INK)
        pdf.polygon(tri, style="F")
        if text:
            pdf.set_font(pdf.body_font, "I", 8)
            pdf.set_text_color(*MUTED)
            pdf.set_xy(*text_at)
            pdf.cell(30, 4, text)

    pill_h = 8
    pdf.set_fill_color(*INK)
    pdf.rect(cx - 15, y0, 30, pill_h, style="F", round_corners=True, corner_radius=4)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(pdf.body_font, "B", 9)
    pdf.set_xy(cx - 15, y0 + 1.8)
    pdf.cell(30, 4.5, "START", align="C")

    y_plan = y0 + pill_h + gap
    y_gen = y_plan + h + gap
    y_exec = y_gen + h + gap
    y_dec = y_exec + h + gap
    dec_h = 16
    y_end = y_dec + dec_h + gap

    arrow([(cx, y0 + pill_h), (cx, y_plan)])
    box(cx, y_plan, "plan", "LLM: 1-2 sentence approach", BOX_FILL)
    arrow([(cx, y_plan + h), (cx, y_gen)])
    box(cx, y_gen, "generate", "LLM: pandas / Plotly code", BOX_FILL)
    arrow([(cx, y_gen + h), (cx, y_exec)])
    box(cx, y_exec, "execute", "sandbox runs code, reads RESULT", BOX_FILL)
    arrow([(cx, y_exec + h), (cx, y_dec)])

    # decision diamond
    dw = 58
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
    fx = cx + 68
    box(fx, y_dec + dec_h / 2 - h / 2, "fix", "LLM: rewrite from traceback", FIX_FILL, width=48)
    arrow([(cx + dw / 2, y_dec + dec_h / 2), (fx - 24, y_dec + dec_h / 2)], "yes", (cx + dw / 2 + 2, y_dec + dec_h / 2 - 5))
    arrow([(fx, y_dec + dec_h / 2 - h / 2), (fx, y_exec + h / 2), (cx + w / 2, y_exec + h / 2)],
          "retries + 1", (fx + 2, y_exec + h + 1))

    # no -> END
    arrow([(cx, y_dec + dec_h), (cx, y_end)], "no", (cx + 2, y_dec + dec_h + 1))
    box(cx, y_end, "END", "answer, or the last error", END_FILL)

    pdf.set_y(y_end + h + 4)


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

    # --- overview
    pdf.add_page()
    pdf.h1("1. Overview")
    pdf.para(
        "The Data Analyst Agent lets someone with no coding skills explore a dataset. They upload a CSV and "
        "type a question such as \"monthly revenue trend\". An LLM plans the analysis, writes pandas or Plotly "
        "code, and runs it in a restricted sandbox. If the code fails, the agent reads the traceback and "
        "rewrites the code, up to three times. The answer appears as a table, a single number, text or an "
        "interactive chart, and the code is always one click away."
    )
    pdf.h2("Key features")
    pdf.bullets([
        "**Natural language to code:** questions become pandas and Plotly code.",
        "**Self-healing:** failed code goes back to the LLM with its traceback, up to 3 fix attempts.",
        "**Automatic visualisation:** the code picks a chart when the question calls for one.",
        "**Schema-aware:** every call carries the column names, types, sample rows and statistics.",
        "**Code transparency:** the plan and the generated code are shown under every answer.",
        "**Follow-ups:** the last five turns, with their code, go into the prompts, so \"now only 2024\" works.",
    ])
    pdf.h2("Results at a glance")
    pdf.bullets([
        f"**{answered} of {len(samples)}** sample questions answered, **{first_try}** of them with the first code.",
        f"The self-correction demo recovered from a real KeyError in **{_attempts(healed['retries'])}**."
        if healed else "No self-correction demo was recorded.",
        f"**{count_tests()}** automated tests cover the loader, prompts, sandbox, agent loop and app (no network needed).",
    ])
    if SCREENSHOT.exists():
        pdf.keep_together(110)
        pdf.label("The Streamlit app")
        pdf.image(str(SCREENSHOT), w=150)
        pdf.ln(3)

    # --- architecture
    pdf.add_page()
    pdf.h1("2. Architecture")
    pdf.data_table(["Layer", "Tool"], [
        ["Frontend", "Streamlit"],
        ["LLM", f"{data['provider']} ({data['model']}); Google Gemini as the alternative"],
        ["Agent framework", "LangGraph state machine"],
        ["Data processing", "pandas, NumPy"],
        ["Charts", "Plotly Express"],
        ["Code execution", "exec() with whitelisted builtins, an AST check and a timeout"],
        ["Report", "fpdf2, with charts exported by Kaleido"],
    ], size=9.5, col_widths=(1, 2.4))
    pdf.h2("Agent graph")
    pdf.para("Every question runs through the same state machine. Each box is a node that reads the shared "
             "state and returns only the keys it changes.")
    pdf.keep_together(120)
    draw_flowchart(pdf)

    pdf.h2("Nodes")
    pdf.data_table(["Node", "What it does"], [
        ["plan", "Asks the LLM for a one or two sentence approach, given the schema, the question and recent history."],
        ["generate", "Asks for Python that uses df, pd, px and np and stores the answer in RESULT. Markdown fences "
                     "and stray prose are stripped from the reply."],
        ["execute", "Runs the code in the sandbox on a copy of the data and classifies RESULT as a table, chart, "
                    "value or text. A failure returns a trimmed traceback."],
        ["fix", "Sends the traceback and the failed code back to the LLM for a corrected version; adds 1 to retries."],
        ["should_retry", "Routes to fix while there is an error and fewer than 3 fixes have been tried; otherwise ends."],
    ], size=9, col_widths=(1, 4))

    pdf.h2("The sandbox")
    pdf.para("Generated code is checked and run with deliberately few powers:")
    pdf.bullets([
        "An **AST check** rejects imports, file access, eval/exec, dunder attributes and namespace tricks before "
        "anything runs, with a message the fix node can act on.",
        "**Whitelisted builtins** only (len, sum, sorted, round, and similar), so open() and the import function do not exist.",
        "The code gets a **copy** of the DataFrame, so a bad in-place edit cannot change the user's data.",
        "A **10-second timeout** stops runaway loops from blocking the app.",
    ])
    pdf.para("This guards against accidents, not attackers: it is not a security boundary.", color=MUTED, style="I")

    pdf.h2("LLM calls")
    pdf.para(
        "One question costs at least two LLM calls (plan and generate), plus one per fix. Temporary failures "
        "(rate limits, 5xx errors, timeouts) are retried up to three times, waiting as long as the provider asks, "
        "and the app tells the user it is waiting. A wait longer than a minute, such as a used-up daily quota, "
        "stops at once with a message saying when the quota resets. Bad keys and unknown model names are reported "
        "without retrying."
    )

    # --- sample outputs
    pdf.add_page()
    pdf.h1("3. Sample outputs")
    pdf.para(f"The spec's sample questions, asked of {dataset['file']} ({dataset['rows']:,} rows, "
             f"{dataset['columns']} columns: date, product, region, revenue, quantity, salesperson). "
             "Each shows the agent's plan, its code and the result.")
    for number, entry in enumerate(samples, 1):
        render_answer(pdf, number, entry)

    # --- self-correction
    if healed:
        pdf.add_page()
        pdf.h1("4. Self-correction in action")
        pdf.para(
            "On clean data the model rarely writes failing code, so the loop is hard to see. For this demo the "
            "first code was replaced with a version that uses a misspelled column (revenu). Everything after "
            "that is real: the sandbox raised a genuine KeyError, and the live model's fix node read the "
            "traceback and repaired the code."
        )
        pdf.label(f"Question: {healed['question']}")
        pdf.ln(1)
        pdf.label("1. First code (broken on purpose)")
        pdf.code(healed["broken_code"])
        pdf.label("2. What the sandbox returned to the fix node")
        pdf.code(healed["first_error"], max_lines=ERROR_MAX_LINES)
        pdf.label(f"3. Code after {_attempts(healed['retries'])}")
        pdf.code(healed["code"])
        if healed.get("rows"):
            pdf.label("4. Result")
            pdf.data_table(healed["columns"], healed["rows"][:TABLE_ROWS])

    # --- limitations and learnings
    pdf.add_page()
    pdf.h1("5. Limitations")
    pdf.bullets([
        "**The sandbox is not a security boundary.** It blocks accidental damage, but exec() inside the app's "
        "process cannot stop a determined attacker. Do not expose the app publicly with untrusted users.",
        "**A timed-out run keeps going in the background.** Python threads cannot be killed, so the 10-second "
        "limit frees the app but not the CPU.",
        "**Wrong but working answers pass.** The retry loop catches code that crashes, not code that answers a "
        "different question. Showing the plan and the code is the safeguard.",
        "**One CSV at a time,** and the LLM sees only a summary of it (schema, sample rows, statistics).",
        "**Free-tier limits.** Groq's free tier limits tokens per minute, so a fast series of questions waits a "
        "few seconds; Gemini's free key allowed only 20 requests a day per model.",
    ])
    pdf.h1("6. Learnings")
    pdf.bullets([
        "**Restricting exec() takes more than a small globals dict.** Python adds the full builtins unless the "
        "code's builtins are replaced, so open() and the import function stay available. A whitelist plus an AST "
        "check closed that.",
        "**Models do not follow format rules reliably.** Replies arrive in markdown fences, with prose around the "
        "code, or with print() calls. Cleaning the reply before running it removed a whole class of failures.",
        "**Errors are prompts.** Sending a short traceback (the last 30 lines) back to the model is usually "
        "enough for it to fix a wrong column name or type in one attempt.",
        "**Follow-ups need the previous code,** not just the previous question. History holds each turn's code "
        "and a one-line summary of its result.",
        "**Free-tier models move.** gemini-2.5-flash closed to new keys and the free quota was 20 requests a day, "
        "which ran out during testing. Keeping the model name in settings, and adding Groq as a second provider "
        "behind the same retry and error handling, kept the project working.",
        "**Test the loop without the network.** A scripted fake LLM made the retry routing, history and error "
        "handling testable in seconds, with no API key.",
    ])
    return pdf


def render_answer(pdf: Report, number: int, entry: dict):
    kind = entry["result_type"]
    pdf.keep_together(min(_answer_height(entry), 240))
    pdf.ln(2)
    pdf.set_font(pdf.body_font, "B", 11.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 6, pdf.text_safe(f"Q{number}. \"{entry['question']}\""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    detail = QUESTION_KINDS.get(kind, "Failed") if not entry["error"] else "Failed"
    if kind == "plotly" and entry.get("chart_type"):
        detail += f" ({entry['chart_type']})"
    fixes = _attempts(entry["retries"])
    pdf.set_font(pdf.body_font, "", 8.5)
    pdf.set_text_color(*MUTED)
    extra = f"  ·  {entry['note']}" if entry.get("note") else ""
    pdf.cell(0, 5, pdf.text_safe(f"{detail}  ·  {fixes}  ·  {entry['seconds']} s{extra}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.para(f"**Plan:** {_plain(entry['plan'])}", size=9.5)
    pdf.code(entry["code"])
    if entry["error"]:
        pdf.label("Last error")
        pdf.code(entry["error"], max_lines=ERROR_MAX_LINES)
    elif kind == "dataframe":
        rows = entry["rows"][:TABLE_ROWS]
        pdf.data_table(entry["columns"], rows)
        if entry["row_count"] > len(rows):
            pdf.para(f"First {len(rows)} of {entry['row_count']:,} rows.", size=8.5, color=MUTED)
    elif kind == "plotly" and (OUTPUTS / entry["chart"]).exists():
        pdf.keep_together(CHART_H)
        pdf.image(str(OUTPUTS / entry["chart"]), w=CHART_W)
        pdf.ln(3)
    elif kind == "value":
        pdf.set_font(pdf.body_font, "B", 16)
        pdf.set_text_color(*ACCENT)
        value = entry["value"]
        pdf.cell(0, 9, f"{value:,.2f}" if isinstance(value, float) else str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
    else:
        pdf.para(_plain(str(entry.get("value", ""))), size=10.5, style="B")
    pdf.ln(2)


def _attempts(n: int) -> str:
    return f"{n} fix attempt{'s' if n != 1 else ''}"


def _answer_height(entry: dict) -> float:
    """A rough height in mm of one answer block, to avoid splitting it across pages."""
    plan_lines = sum(len(line) // 105 + 1 for line in _plain(entry["plan"]).splitlines())
    code_lines = min(len(entry["code"].splitlines()), CODE_MAX_LINES + 1)
    height = 20 + plan_lines * 5 + code_lines * 4.1 + 8
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
