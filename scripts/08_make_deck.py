"""Build deck/triage_deck.pptx: the project, its novelty, results, and comparison with the literature.

Every number on a slide comes from paper/macros.json, paper/numbers.json, paper/*.csv or the results/
directories that scripts/07_paper_assets.py reads, so the deck cannot drift from the paper. Citation numbers
follow the paper's bibliography order (paper/main.bbl). Run scripts/07_paper_assets.py and compile the paper
(tectonic --keep-intermediates paper/main.tex) first.

    python scripts/08_make_deck.py
"""
import glob
import json
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lxml import etree  # noqa: E402
from PIL import ImageFont  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION  # noqa: E402
from pptx.enum.dml import MSO_LINE  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                      # timelines pickle references src.* classes
PAPER, DECK = os.path.join(ROOT, "paper"), os.path.join(ROOT, "deck")
FIGS = os.path.join(DECK, "figures")
os.makedirs(FIGS, exist_ok=True)

FONT = "Arial"
NAVY, INK, MUTED, LIGHT, LINE = "0F2A47", "1F2937", "5B6573", "F2F5F9", "D5DCE5"
BLUE, ORANGE, GREEN, AMBER, PINK = "2A78D6", "D9531E", "12906A", "EDA100", "E87BA4"
BLUE_T, ORANGE_T, GREEN_T, AMBER_T = "E8F1FC", "FDEEE7", "E5F6EF", "FFF3CC"
WHITE = "FFFFFF"

M = json.load(open(os.path.join(PAPER, "macros.json")))
N = json.load(open(os.path.join(PAPER, "numbers.json")))
SUMM = pd.read_csv(os.path.join(PAPER, "summary_clipped.csv"))
FRONT = pd.read_csv(os.path.join(PAPER, "frontier_all_seeds.csv"))
REPLAY = sorted(glob.glob(os.path.join(ROOT, "results", "run_*_replay_tls-year22")))[-1]
QUIC = sorted(glob.glob(os.path.join(ROOT, "results", "run_*_quic_quic22")))[-1]
TLS_TL = pickle.load(open(os.path.join(REPLAY, "timelines_seed0.pkl"), "rb"))["timelines"]
QUIC_TL = pickle.load(open(os.path.join(QUIC, "timelines_seed0.pkl"), "rb"))["timelines"]
CAL0 = json.load(open(os.path.join(REPLAY, "calibration.json")))[0]


def rgb(h):
    return RGBColor.from_string(h)


# ------------------------------------------------------------------------------ citations
KEYS = re.findall(r"bibitem\{(\w+)\}", open(os.path.join(PAPER, "main.bbl")).read())
NUM = {k: i + 1 for i, k in enumerate(KEYS)}
USED: set[str] = set()


def c(*keys):
    """'[4]' or '[6], [7]' in the paper's numbering; records the keys for the reference slide."""
    USED.update(keys)
    return ", ".join(f"[{NUM[k]}]" for k in sorted(keys, key=NUM.get))


# ------------------------------------------------------------------------------ figures
def _rc():
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#" + MUTED, "axes.linewidth": 0.8,
                         "grid.color": "#E3E8EE", "grid.linewidth": 0.7, "figure.dpi": 200,
                         "axes.labelcolor": "#" + INK, "xtick.color": "#" + INK, "ytick.color": "#" + INK})


def rows(tl, name):
    return pd.DataFrame(tl[name]["rows"])


def fig_decay(path):
    df = rows(TLS_TL, "P0_never")
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.grid(True, axis="y")
    ax.axhline(float(M["pzeroref"]), color="#" + MUTED, lw=1, ls=":")
    ax.text(51.5, float(M["pzeroref"]) + 0.006, f"held-out F1 at deployment: {M['pzeroref']}", ha="right", color="#" + MUTED, fontsize=11)
    ax.plot(df.week, df.macro_f1, color="#" + BLUE, lw=2.6)
    ax.axvline(10, color="#" + ORANGE, lw=1.2, ls="--")
    ax.text(10.6, 0.612, "week 10:\nexporter update", color="#" + ORANGE, fontsize=11, va="bottom")
    ax.annotate(f"{M['pzeromin']} in week 52", xy=(52, float(M["pzeromin"])), xytext=(38, 0.595), color="#" + BLUE, fontsize=12,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color="#" + BLUE, lw=0.8))
    ax.set_ylim(0.58, 0.885)
    ax.set_xlim(2, 53)
    ax.set_xlabel("Week of 2022")
    ax.set_ylabel("Weekly macro-F1 (never retrained)")
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_tls(path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.9, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
    for a in (a1, a2):
        a.grid(True, axis="y")
    for name, col, lab, ls, lw in (("P0_never", MUTED, "Never retrain (P0)", (0, (3, 2)), 1.8),
                                   ("P2_weekly", BLUE, "Retrain weekly (P2)", "-", 1.8),
                                   ("P4c_cumulative", AMBER, "Our cumulative rule (P4c)", "-", 2.4)):
        df = rows(TLS_TL, name)
        a1.plot(df.week, df.macro_f1, color="#" + col, ls=ls, lw=lw, label=lab)
    r4 = rows(TLS_TL, "P4c_cumulative")
    lo = 0.585
    a1.scatter(r4[r4.retrain].week, [lo] * int(r4.retrain.sum()), marker="^", s=55, color="#" + AMBER, edgecolor="white", linewidth=0.6,
               zorder=5, clip_on=False)
    a1.set_ylim(0.57, 0.9)
    a1.set_ylabel("Weekly macro-F1")
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.0)
    p0 = rows(TLS_TL, "P0_never")
    a2.plot(p0.week, p0.D, color="#" + NAVY, lw=2)
    a2.set_ylabel("Drift score D")
    a2.set_xlabel("Week of 2022   (triangles: when P4c retrained)")
    for a in (a1, a2):
        a.axvline(10, color="#" + ORANGE, lw=1.1, ls="--")
    fig.tight_layout(pad=0.4, h_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_pareto(path):
    ratio = 10
    s = SUMM[SUMM.cost_ratio == ratio]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.grid(True)
    fr = FRONT.groupby("budget").mean_f1.mean()
    fr = fr[fr.index <= 30]
    ax.plot(fr.index, fr.values, color="#" + INK, lw=1.4, ls="--", label="Best possible (hindsight, DP)", zorder=2)
    spec = [("periodic", "Periodic calendars", BLUE, "o"), ("P3", "Threshold alarm", ORANGE, "s"), ("P4", "Myopic cost rule", GREEN, "^"),
            ("P4c", "Cumulative cost rule (ours)", AMBER, "D"), ("P5", "Threshold oracle (reads labels)", PINK, "v")]
    for fam, lab, col, mk in spec:
        g = s[s.family == fam].groupby("policy").agg(x=("n_retrains", "mean"), y=("mean_macro_f1", "mean")).sort_values("x")
        g = g[g.x <= 30]
        ax.plot(g.x, g.y, color="#" + col, marker=mk, ms=6.5 if fam != "P4c" else 8, lw=1.6 if fam != "P4c" else 2.6, mec="white", mew=0.6,
                label=lab, zorder=3 if fam != "P4c" else 4)
    pt = lambda p: s[s.policy == p].agg({"n_retrains": "mean", "mean_macro_f1": "mean"})
    for p, lab, off in (("P0_never", "never", (8, -4)), ("P1_monthly", "monthly", (6, -14)), ("P4c_cumulative", "P4c", (8, -16)),
                        ("P4_proposed", "P4", (6, -18))):
        v = pt(p)
        ax.annotate(lab, (v.n_retrains, v.mean_macro_f1), xytext=off, textcoords="offset points", fontsize=11, color="#" + INK, fontweight="bold")
    ax.set_xlabel("Retrains over the 50 replay weeks")
    ax.set_ylabel("Mean weekly macro-F1")
    ax.set_xlim(-0.8, 30)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_quic(path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
    for a in (a1, a2):
        a.grid(True, axis="y")
    for name, col, lab, ls, lw in (("P0_never", MUTED, "Never retrain", (0, (3, 2)), 1.8), ("P5_oracle", PINK, "Oracle (reads labels)", "-", 1.8),
                                   ("P4c_cumulative", AMBER, "Our rule (P4c)", "-", 2.6)):
        df = rows(QUIC_TL, name)
        a1.plot(df.week, df.macro_f1, color="#" + col, ls=ls, lw=lw, label=lab)
    a1.set_ylim(0.585, 0.725)
    for name, col, y in (("P4c_cumulative", AMBER, 0.593), ("P5_oracle", PINK, 0.603)):
        r = rows(QUIC_TL, name)
        a1.scatter(r[r.retrain].week, [y] * int(r.retrain.sum()), marker="^", s=55, color="#" + col, edgecolor="white", linewidth=0.6, zorder=5)
    a1.set_ylabel("Daily macro-F1")
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.0)
    p0 = rows(QUIC_TL, "P0_never")
    a2.plot(p0.week, p0.D, color="#" + NAVY, lw=2)
    a2.set_ylabel("Drift score D")
    a2.set_xlabel("Day of replay   (triangles: retrains)")
    from matplotlib.ticker import MaxNLocator
    a2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ev = N["quic"]["event_day"]
    for a in (a1, a2):
        a.axvline(ev, color="#" + ORANGE, lw=1.3, ls="--")
    a1.text(ev + 0.3, 0.72, "certificate change\n(day 22)", color="#" + ORANGE, fontsize=10.5, va="top")
    fig.tight_layout(pad=0.4, h_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def fig_cost(path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.7, 4.3), sharex=True)
    wk = np.arange(1, 11)
    w = 0.22                                    # excess-loss cost per week, in units of the cost of fixing
    a1.bar(wk, [w] * 10, color="#" + MUTED, width=0.62)
    a1.axhline(1.0, color="#" + ORANGE, lw=1.6, ls="--")
    a1.text(10.5, 1.04, "cost of fixing", color="#" + ORANGE, ha="right", fontsize=12.5)
    a1.set_ylim(0, 1.45)
    a1.set_title("Myopic rule: one week's loss vs. the fixed cost", fontsize=13.5, loc="left", color="#" + INK)
    a1.text(5.5, 0.52, "never reaches the line, so it never retrains", ha="center", color="#" + MUTED, fontsize=12.5)
    run = 0.0
    heights = []
    for i in range(10):
        run += w
        heights.append(run)
        if run > 1.0:
            run = 0.0
    a2.bar(wk, heights, color="#" + AMBER, width=0.62)
    a2.axhline(1.0, color="#" + ORANGE, lw=1.6, ls="--")
    cross = next(i + 1 for i, h in enumerate(heights) if h > 1.0)
    a2.annotate("retrain", (cross, heights[cross - 1]), xytext=(cross - 2.3, 1.28), fontsize=13.5, fontweight="bold", color="#" + INK,
                arrowprops=dict(arrowstyle="->", color="#" + INK))
    a2.set_ylim(0, 1.45)
    a2.set_title("Cumulative rule (ours): running total of excess loss", fontsize=13.5, loc="left", color="#" + INK)
    a2.set_xlabel("Weeks since the model was deployed (illustrative)")
    for a in (a1, a2):
        a.set_yticks([])
        a.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.4, h_pad=0.8)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------ pptx helpers
FONT_FILE = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WARN: list[str] = []


def _measure(text, size, bold=False):
    f = ImageFont.truetype(FONT_BOLD if bold else FONT_FILE, int(size * 10))
    return f.getlength(text) / 10.0


def lines_needed(text, size, width_pt, bold=False):
    n = 0
    for para in text.split("\n"):
        words, cur, k = para.split(" "), "", 1
        for wd in words:
            t = (cur + " " + wd).strip()
            if _measure(t, size, bold) <= width_pt:
                cur = t
            else:
                k += 1
                cur = wd
        n += k
    return n


def split_runs(text):
    parts = text.split("**")
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p != ""]


def flat(sh):
    """No drop shadow: clear the effect list and neutralise the theme's effect reference."""
    sh.shadow.inherit = False
    st = sh._element.find(qn("p:style"))
    if st is not None:
        er = st.find(qn("a:effectRef"))
        if er is not None:
            er.set("idx", "0")


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.layout = self.prs.slide_layouts[5]
        self.n = 0
        self.cur = ""

    # -- text
    def text(self, slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             line=1.1, space=4, inset=0.08, check=True):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(inset)
        tf.margin_top = tf.margin_bottom = Inches(0.04)
        if isinstance(paras, str):
            paras = [paras]
        need = 0.0
        for i, p in enumerate(paras):
            spec = p if isinstance(p, dict) else {"t": p}
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            sz = spec.get("size", size)
            para.alignment = spec.get("align", align)
            para.line_spacing = line
            para.space_after = Pt(spec.get("space", space))
            bul = spec.get("bullet", False)
            txt_all = ""
            for t, b in split_runs(spec["t"]):
                r = para.add_run()
                r.text = t
                r.font.size, r.font.name = Pt(sz), FONT
                r.font.bold = bold or spec.get("bold", False) or b
                r.font.italic = spec.get("italic", False)
                r.font.color.rgb = rgb(spec.get("color", color))
                txt_all += t
            avail = (w - 2 * inset) * 72 - (16 if bul else 0)
            if bul:
                pPr = para._p.get_or_add_pPr()
                pPr.set("marL", str(int(Pt(16)))); pPr.set("indent", str(-int(Pt(16))))
                bu = etree.SubElement(pPr, qn("a:buFont")); bu.set("typeface", "Arial")
                bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•")
            n = lines_needed(txt_all, sz, avail, bold or spec.get("bold", False))
            need += n * sz * 1.2 * line + (spec.get("space", space) if i < len(paras) - 1 else 0)
        if check and need / 72 > h - 0.02:
            WARN.append(f"slide {self.n} ('{self.cur[:38]}'): text needs {need / 72:.2f} in, box is {h:.2f} in: {str(paras[0])[:50]}")
        return tb

    def card(self, slide, x, y, w, h, fill=LIGHT, accent=None, line=None, radius=0.05):
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.adjustments[0] = radius
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
        if line:
            sh.line.color.rgb = rgb(line); sh.line.width = Pt(1)
        else:
            sh.line.fill.background()
        flat(sh)
        if accent:
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.1), Inches(0.07), Inches(h - 0.2))
            bar.fill.solid(); bar.fill.fore_color.rgb = rgb(accent); bar.line.fill.background(); flat(bar)
        return sh

    def box(self, slide, x, y, w, h, text, fill, color=WHITE, size=14, bold=True, dashed=False, line=None, radius=0.12):
        sh = self.card(slide, x, y, w, h, fill=fill, line=line, radius=radius)
        if dashed:
            sh.line.color.rgb = rgb(MUTED); sh.line.width = Pt(1.5); sh.line.dash_style = MSO_LINE.DASH
        tf = sh.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.06)
        for i, ln in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            for t, b in split_runs(ln):
                r = p.add_run(); r.text = t
                r.font.size, r.font.name, r.font.bold = Pt(size), FONT, (bold or b)
                r.font.color.rgb = rgb(color)
        need = sum(lines_needed(re.sub(r"\*\*", "", ln), size, (w - 0.12) * 72, bold) for ln in text.split("\n")) * size * 1.2 / 72
        if need > h - 0.05:
            WARN.append(f"slide {self.n}: box text needs {need:.2f} in vs {h:.2f}: {text[:40]!r}")
        return sh

    def arrow(self, slide, x1, y1, x2, y2, color=MUTED, dashed=False, width=1.75, head=True):
        cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        cn.line.color.rgb = rgb(color); cn.line.width = Pt(width)
        if dashed:
            cn.line.dash_style = MSO_LINE.DASH
        if head:
            ln = cn.line._get_or_add_ln()
            te = etree.SubElement(ln, qn("a:tailEnd")); te.set("type", "triangle")
        return cn

    def pic(self, slide, path, x, y, w=None, h=None, alt=""):
        p = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
        p._element.nvPicPr.cNvPr.set("descr", alt)
        return p

    def table(self, slide, x, y, w, col_w, data, size=12, row_h=0.4, header=True, hl_rows=(), fills=None, bold_cols=(), aligns=None,
              head_size=None, head_h=None):
        nr, nc = len(data), len(data[0])
        gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        tbl = gf.table
        tbl.horz_banding = False
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(w * cw / tot)
        for i in range(nr):
            tbl.rows[i].height = Inches(head_h if (head_h and header and i == 0) else row_h)
            for j in range(nc):
                cell = tbl.cell(i, j)
                val = data[i][j]
                fill = NAVY if (header and i == 0) else (fills.get((i, j)) if fills and (i, j) in fills else (AMBER_T if i in hl_rows else (WHITE if i % 2 else LIGHT)))
                cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = cell.margin_right = Inches(0.07)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
                txt = ""
                for t, b in split_runs(str(val)):
                    r = p.add_run(); r.text = t
                    sz = head_size if (header and i == 0 and head_size) else size
                    r.font.size, r.font.name = Pt(sz), FONT
                    r.font.bold = b or (header and i == 0) or (j in bold_cols) or (i in hl_rows and j == 0)
                    r.font.color.rgb = rgb(WHITE if (header and i == 0) else INK)
                    txt += t
                avail = (w * col_w[j] / tot - 0.14) * 72
                n = lines_needed(txt, size, avail, bold=(header and i == 0))
                if n * size * 1.2 / 72 + 0.06 > row_h + 0.001 and n > 1:
                    WARN.append(f"slide {self.n}: table cell wraps to {n} lines (row_h {row_h}): {txt[:40]!r}")
        return gf

    # -- slides
    def slide(self, title, notes="", dark=False):
        self.n += 1
        self.cur = title
        s = self.prs.slides.add_slide(self.layout)
        if dark:
            s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(NAVY)
        t = s.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.13), Inches(1.05)
        tf = t.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.08)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = title
        r.font.size, r.font.name, r.font.bold = Pt(28 if not dark else 44), FONT, True
        r.font.color.rgb = rgb(WHITE if dark else NAVY)
        if lines_needed(title, 28, (12.13 - 0.2) * 72, True) > 2 and not dark:
            WARN.append(f"slide {self.n}: title wraps to >2 lines")
        if not dark:
            ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.68), Inches(1.4), Inches(0.9), Inches(0.05))
            ln.fill.solid(); ln.fill.fore_color.rgb = rgb(AMBER); ln.line.fill.background(); flat(ln)
            self.text(s, 0.6, 7.02, 9.0, 0.3, "When to retrain: label-free, cost-aware drift triage", size=10, color=MUTED, check=False)
            self.text(s, 11.9, 7.02, 0.83, 0.3, str(self.n), size=10, color=MUTED, align=PP_ALIGN.RIGHT, check=False)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s


def num(x, d=3):
    return f"{float(x):.{d}f}"


def build(path):
    _rc()
    fig_decay(os.path.join(FIGS, "decay.png"))
    fig_tls(os.path.join(FIGS, "tls.png"))
    fig_pareto(os.path.join(FIGS, "pareto.png"))
    fig_quic(os.path.join(FIGS, "quic.png"))
    fig_cost(os.path.join(FIGS, "cost.png"))
    D = Deck()
    f1 = lambda k: N[k]["f1"]
    d_f1 = N["P4c_cumulative"]["f1"] - N["P4_proposed"]["f1"]
    d_ret = N["P4c_cumulative"]["retrains"] - N["P4_proposed"]["retrains"]
    exp_drop = (float(M["pzerowkNine"]) - float(M["pzerowkEleven"])) * 100
    ck = N["checkpoint"]
    sig = ck["per_signal_weekly_level"]

    # ---- 1 title
    s = D.slide("When to retrain", dark=True, notes=(
        "Opening: this project decides WHEN a deployed encrypted-traffic classifier should be retrained, without labels, "
        "weighing the cost of retraining against the cost of staying degraded. Everything in this deck is produced by scripts in the repository."))
    s.shapes.title.top, s.shapes.title.height = Inches(1.9), Inches(1.2)
    D.text(s, 0.6, 3.1, 11.5, 1.2, "Label-free, cost-aware drift triage for encrypted traffic classifiers", size=26, color=WHITE, check=False)
    D.text(s, 0.6, 4.75, 11.5, 0.5, "Lavnish J. (24BCE2127)   ·   Darsh S. (24BDS0082)   ·   Dhruv M. (24BKT0116)", size=18, color="C9D6E6", check=False)
    D.text(s, 0.6, 5.35, 11.5, 0.45, "Evaluated on CESNET-TLS-Year22 (a full year) and CESNET-QUIC22, five random seeds", size=14, color="9FB2CA", check=False)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.76), Inches(4.45), Inches(1.2), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(AMBER); bar.line.fill.background(); flat(bar)

    # ---- 2 problem
    s = D.slide("A classifier that is accurate in week 1 is not accurate in week 52", notes=(
        f"This is the never-retrained model from our experiment: trained on weeks 1-2 of 2022, held-out macro-F1 {M['pzeroref']}, "
        f"falling to {M['pzeromin']} by week 52. There is a step at the week-10 flow-exporter update and slow erosion afterwards. "
        "Seed 0; the other seeds look the same (mean weekly F1 0.709). Source: results/run_*_ref_tls-year22 and replay run."))
    D.pic(s, os.path.join(FIGS, "decay.png"), 0.6, 1.65, w=6.4, alt=f"Line chart: weekly macro-F1 of a never-retrained classifier falls from {M['pzeroref']} to {M['pzeromin']} over 2022")
    D.text(s, 0.6, 5.4, 6.4, 0.5, "Classifier trained on weeks 1–2 of 2022, never retrained (50 services, seed 0).", size=11, color=MUTED)
    items = [("Traffic drifts", f"Apps update, certificates rotate, exporters change. Never-retrained macro-F1: **{M['pzeroref']} → {M['pzeromin']}**.", ORANGE),
             ("Labels are scarce", f"QUIC and Encrypted ClientHello hide the fields used to label flows, so a deployed model is hard to grade {c('luxemburk2023', 'shamsimukhametov2022')}.", BLUE),
             ("Retraining is not free", "Each retrain needs fresh labels and compute, so retraining weekly “just in case” wastes both.", GREEN)]
    for i, (h, b, col) in enumerate(items):
        y = 1.65 + i * 1.55
        D.card(s, 7.3, y, 5.43, 1.42, accent=col)
        D.text(s, 7.5, y + 0.06, 5.1, 0.35, h, size=16, bold=True, color=NAVY)
        D.text(s, 7.5, y + 0.42, 5.1, 0.98, b, size=13.5)
    D.card(s, 0.6, 6.3, 12.13, 0.6, fill=NAVY)
    D.text(s, 0.8, 6.33, 11.8, 0.55, "Question: without labels, how can an operator tell when drift is bad enough to justify the price of retraining?",
           size=16, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 3 background
    s = D.slide("Three ideas you need: classify, drift, retrain", notes=(
        "Plain-language background. Classification: payloads are encrypted, so models use flow statistics. "
        "Drift: the traffic after deployment no longer looks like the training traffic. Retraining: refit on recent labeled flows. "
        "Our task has 50 services and 60 flow features per flow (packet sizes, timings, flags, histograms)."))
    cards = [("1", "Classify encrypted traffic", "Payloads are unreadable, so models use **flow statistics**: packet sizes, timings, directions. The task here: name the service (Teams? Spotify?) among **50 services**, from **60 features** per flow.", BLUE),
             ("2", "Concept drift", "After deployment the traffic changes: app updates, new certificates, a changed flow exporter. Yesterday's patterns stop matching and **accuracy falls**.", ORANGE),
             ("3", "Retrain", "Refit the model on **recent labeled flows**. This restores accuracy, but labels and compute cost money, so it should happen only when it pays off.", GREEN)]
    for i, (n_, h, b, col) in enumerate(cards):
        x = 0.6 + i * 4.1
        D.card(s, x, 1.75, 3.93, 3.75, fill=LIGHT)
        D.box(s, x + 0.2, 1.95, 0.6, 0.6, n_, fill=col, size=22, radius=0.5)
        D.text(s, x + 0.9, 1.95, 2.9, 0.65, h, size=17, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        D.text(s, x + 0.2, 2.8, 3.55, 2.65, b, size=16.5)
    D.card(s, 0.6, 5.85, 12.13, 0.85, fill=AMBER_T, accent=AMBER)
    D.text(s, 0.85, 5.88, 11.8, 0.8, "Our job: decide **when** to retrain, using only what the model itself can observe (its outputs and its inputs), and weigh that against what retraining costs.",
           size=16, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 4 literature gap
    s = D.slide("What the literature offers, and the gap it leaves", notes=(
        "Three strands: measurement of drift in encrypted traffic, label-free drift detection, and retraining policies. "
        "Each answers part of the question; none is a label-free, cost-aware trigger for encrypted-traffic classifiers. "
        "Reference numbers match the paper's bibliography."))
    cols = [("Drift in encrypted traffic " + c("luxemburk2023", "kurapov2024", "jancicka2024", "malekghaini2023"),
             f"Measures it: 9–13 accuracy points lost within three weeks on QUIC {c('luxemburk2023')}; monthly retraining suffices {c('kurapov2024')}; weekly seasonality {c('jancicka2024')}; damage depends on features {c('malekghaini2023')}.",
             "Stops short of: when a shift justifies retraining.", BLUE),
            ("Label-free drift detection " + c("yang2022", "lughofer2016", "sethi2017", "souza2020"),
             f"Jensen–Shannon divergence {c('yang2022')}; classifier confidence {c('lughofer2016')}; margin density {c('sethi2017')}; high-speed streams {c('souza2020')}.",
             "Stops short of: treating “something changed” as the end goal.", GREEN),
            ("Retraining policies " + c("gudepu2023", "kanda2023", "bagui2025", "liu2023oal", "cai2024"),
             f"Outlier-triggered retraining {c('gudepu2023')}; cost-optimal policy, but its detector reads labels {c('kanda2023')}; Isolation-Forest loop on security data {c('bagui2025')}; active learning {c('liu2023oal')}; incremental update {c('cai2024')}.",
             "Stops short of: label-free + cost-aware + encrypted traffic.", ORANGE)]
    for i, (h, b, stop, col) in enumerate(cols):
        x = 0.6 + i * 4.1
        D.card(s, x, 1.75, 3.93, 3.95, fill=LIGHT, accent=col)
        D.text(s, x + 0.2, 1.82, 3.65, 0.65, h, size=15, bold=True, color=NAVY)
        D.text(s, x + 0.2, 2.5, 3.65, 2.15, b, size=14.5)
        D.text(s, x + 0.2, 4.72, 3.65, 0.95, stop, size=14, bold=True, color=col)
    D.card(s, 0.6, 5.95, 12.13, 0.9, fill=NAVY)
    D.text(s, 0.8, 5.97, 11.8, 0.86, "**The gap:** a label-free trigger for encrypted-traffic classifiers that prices expected loss against retraining cost, and is judged against the best schedule hindsight allows.",
           size=15.5, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 5 our idea
    s = D.slide("Our idea: a monitor that sits beside the classifier", notes=(
        "The classifier is a fixed substrate: we never touch it. The monitor reads only its probability output and the raw input features, "
        "turns them into one drift score each day, filters out noise, prices the expected loss, and decides weekly whether to retrain. "
        "Labels enter at exactly two points: one-time calibration of the loss map g, and when a retrain actually fires."))
    labels = [("Weekly\nflows", BLUE), ("Frozen\nclassifier\n(LightGBM)", BLUE), ("Observer:\n8 signals\n→ score D", GREEN),
              ("Change\ndetection\nSTL · PELT\npersistence", GREEN), ("Cost rule:\nloss vs.\ncost of fixing", GREEN), ("Retrain on\nlast 2 weeks\nof labels", ORANGE)]
    bw, gap, x0, y0 = 1.72, 0.36, 0.6, 2.0
    for i, (t, col) in enumerate(labels):
        x = x0 + i * (bw + gap)
        D.box(s, x, y0, bw, 1.55, t, fill=col, size=14)
        if i < len(labels) - 1:
            D.arrow(s, x + bw + 0.02, y0 + 0.78, x + bw + gap - 0.02, y0 + 0.78, color=NAVY, width=2.25)
    sx, sy, sw, sh_ = x0 + 3.1 * (bw + gap), y0 + 2.3, 3.55, 0.8
    D.box(s, sx, sy, sw, sh_, "Sealed label store\nopens only for calibration and retraining", fill=WHITE, color=INK, size=13, bold=False, dashed=True)
    cost_cx = x0 + 4 * (bw + gap) + bw / 2
    ret_cx = x0 + 5 * (bw + gap) + bw / 2
    D.arrow(s, cost_cx, sy, cost_cx, y0 + 1.57, color=ORANGE, dashed=True)                    # calibration of g (once)
    D.arrow(s, sx + sw, sy + sh_ / 2, ret_cx, sy + sh_ / 2, color=ORANGE, dashed=True, head=False)
    D.arrow(s, ret_cx, sy + sh_ / 2, ret_cx, y0 + 1.57, color=ORANGE, dashed=True)            # labels for a fired retrain
    D.text(s, cost_cx - 1.0, y0 + 1.6, 1.0, 0.3, "calibrate g", size=10, color=ORANGE, align=PP_ALIGN.RIGHT, check=False)
    D.text(s, ret_cx + 0.05, y0 + 1.95, 0.9, 0.3, "at retrain", size=10, color=ORANGE, check=False)
    D.text(s, 0.6, y0 + 1.62, 6.0, 0.4, "green = label-free monitor   ·   blue = existing system   ·   orange = uses labels", size=11, color=MUTED)
    props = [("Label-free", "reads only probabilities and raw features"), ("Cost-aware", "retrains when it pays off, not when an alarm rings"),
             ("Judged against the optimum", "compared with the best schedule hindsight allows")]
    for i, (h, b) in enumerate(props):
        x = 0.6 + i * 4.1
        D.card(s, x, 5.35, 3.93, 1.35, fill=LIGHT, accent=AMBER)
        D.text(s, x + 0.2, 5.4, 3.6, 0.4, h, size=16, bold=True, color=NAVY)
        D.text(s, x + 0.2, 5.85, 3.6, 0.8, b, size=14)

    # ---- 6 observer
    s = D.slide("Step 1: eight label-free signals become one drift score", notes=(
        "Signals are computed daily from a 10,000-flow sample. Three design choices matter: orientation (so every signal rises when things get worse), "
        "family-balanced weights (four of the eight signals are confidence statistics and are strongly collinear), and a per-model baseline "
        "(so D means the same thing after a retrain). Nothing in the score is learned."))
    data = [["Family", "Signals", "What it senses"],
            ["Confidence", "max probability (mean, 10th pct.), top-2 margin, entropy", "model is less sure of itself"],
            ["Prediction shift", "Jensen–Shannon divergence of predicted-class histogram", "different mix of predicted services"],
            ["Input drift", "KS statistic, kernel MMD²", "the features themselves moved"],
            ["Reject rate", "share of flows below a confidence cutoff", "more “don't know”"]]
    D.table(s, 0.6, 1.75, 7.3, [1.5, 3.6, 2.4], data, size=12.5, row_h=0.64, bold_cols=(0,), head_h=0.45)
    D.card(s, 0.6, 5.0, 7.3, 1.8, fill=LIGHT, accent=BLUE)
    D.text(s, 0.85, 5.05, 6.9, 0.35, "The score", size=14, bold=True, color=NAVY)
    D.text(s, 0.85, 5.42, 6.9, 1.35, ["**D = Σ wᵢ · σᵢ · (sᵢ − bᵢ) / ŝᵢ**",
                                      "sᵢ today's signal  ·  bᵢ its value for this model when trained  ·  ŝᵢ normal daily spread",
                                      "σᵢ = ±1 orientation  ·  wᵢ family weight"], size=13, space=5)
    tips = [("Orient", "Confidence falls under drift while entropy rises. Flipping signs makes every signal “higher = worse”."),
            ("Balance", "Four of eight signals are confidence statistics. Each family gets ¼ of the weight, so they cannot outvote the rest."),
            ("Re-baseline", "Each model generation has its own baseline, so D keeps its meaning after a retrain.")]
    for i, (h, b) in enumerate(tips):
        y = 1.75 + i * 1.7
        D.card(s, 8.2, y, 4.53, 1.55, fill=LIGHT, accent=GREEN)
        D.text(s, 8.4, y + 0.04, 4.2, 0.35, h, size=15, bold=True, color=NAVY)
        D.text(s, 8.4, y + 0.42, 4.2, 1.1, b, size=12.5)

    # ---- 7 change detection
    sweep = CAL0["pelt_penalty_sweep_calm"]
    s = D.slide("Step 2: ignore noise, act only on persistent shifts", notes=(
        "Three stages on the daily D series. STL removes the day-of-week cycle. PELT finds change points; its penalty is the smallest value on a grid "
        f"that gives zero change points on the calm weeks 3-8, which here is {int(CAL0['pelt_penalty'])}. The right-hand chart is the measured sweep, seed 0: penalties 1-5 produced "
        "3-4 false change points on the calm window, 8 and above produced none. A persistence gate then requires the shift to be upward, to last 3 days and to be still active. "
        "The chain is causal: a test perturbs future data and checks past detections do not change."))
    st = [("Remove the weekly cycle", "Robust STL (period 7) on the daily score D."), ("Find change points", "PELT with an ℓ₂ cost. Penalty = smallest value that gives zero change points on a calm stretch."),
          ("Require persistence", "Upward shift, stays above the old level for k = 3 days, and is still active today. Then it is a drift event.")]
    for i, (h, b) in enumerate(st):
        y = 1.75 + i * 1.4
        D.box(s, 0.6, y, 0.6, 0.6, str(i + 1), fill=GREEN, size=20, radius=0.5)
        D.text(s, 1.35, y - 0.05, 5.6, 0.4, h, size=16, bold=True, color=NAVY)
        D.text(s, 1.35, y + 0.37, 5.6, 0.95, b, size=14.5)
    D.card(s, 0.6, 6.0, 6.35, 0.8, fill=BLUE_T)
    D.text(s, 0.8, 6.07, 6.0, 0.72, "Causal by construction: a test changes the future and checks that past detections stay the same.", size=12.5, anchor=MSO_ANCHOR.MIDDLE)
    pens = [1, 2, 3, 5, 8, 12, 20]
    cd = CategoryChartData(); cd.categories = [str(p) for p in pens]; cd.add_series("false change points", [sweep[str(float(p))] if str(float(p)) in sweep else sweep[str(p)] for p in pens])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.3), Inches(2.15), Inches(5.43), Inches(3.55), cd)
    ch = gf.chart
    ch.has_legend = False; ch.has_title = False; ch.font.size = Pt(12); ch.font.name = FONT
    pl = ch.plots[0]; pl.gap_width = 55; pl.has_data_labels = True
    pl.data_labels.font.size = Pt(13); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.data_labels.number_format = "0"; pl.data_labels.number_format_is_linked = False
    for i, p in enumerate(pens):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(GREEN if sweep.get(str(float(p)), sweep.get(str(p), 1)) == 0 else MUTED)
    ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False; ch.value_axis.maximum_scale = 5
    ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "PELT penalty"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    D.text(s, 7.3, 1.7, 5.43, 0.45, "False change points on the calm weeks 3–8", size=14, bold=True, color=NAVY)
    D.text(s, 7.3, 5.75, 5.43, 0.9, f"Penalties 1–5 raise false alarms; we take the smallest penalty with none (λ = {int(CAL0['pelt_penalty'])}). This is the trade-off Smirnov describes {c('smirnov2023')}.", size=12, color=MUTED)

    # ---- 8 cost rule
    s = D.slide("Step 3: is the damage worth the price of fixing it?", notes=(
        "g is an isotonic map from D to expected macro-F1 loss, fitted once on weeks 3-10 and then frozen. The myopic rule (the specification's rule) compares ONE week of expected loss "
        "with the one-off cost of a retrain: a steady moderate loss never trips it even though its total cost grows without bound. Our cumulative rule adds up the excess loss since the "
        "model was deployed and retrains when the total passes the cost of fixing. The right-hand chart is a schematic, not data. The single operator knob is the cost ratio r."))
    cards = [("Loss map g", "Maps the score D to expected F1 loss. Isotonic regression, fitted once on weeks 3–10, then frozen.", BLUE),
             ("Myopic rule (P4)", "Retrain if **g(D) · volume · cost of error > cost of fixing**. Compares one week of loss with a one-off price: a steady, moderate loss never trips it.", ORANGE),
             ("Cumulative rule (P4c, ours)", "Add up the **excess loss since the model was deployed**. Retrain when the total passes the cost of fixing, then reset.", AMBER),
             ("One knob", "r = cost of error ÷ cost of fixing. We sweep r from 0.3 to 1000.", GREEN)]
    ys = [1.7, 2.75, 4.0, 5.3]
    hs = [0.95, 1.18, 1.18, 0.95]
    for (h, b, col), y, hh in zip(cards, ys, hs):
        D.card(s, 0.6, y, 6.6, hh, fill=LIGHT, accent=col)
        D.text(s, 0.8, y + 0.03, 6.3, 0.35, h, size=14.5, bold=True, color=NAVY)
        D.text(s, 0.8, y + 0.36, 6.3, hh - 0.38, b, size=12.5)
    D.pic(s, os.path.join(FIGS, "cost.png"), 7.5, 1.75, w=5.23, alt="Schematic: with the same weekly loss the myopic rule never reaches the cost of fixing, while the cumulative rule's running total crosses it and triggers a retrain")
    D.text(s, 7.5, 5.75, 5.23, 0.9, "Schematic with the same loss every week: the myopic rule never fires, the cumulative rule fires once the total passes the cost of fixing.", size=11.5, color=MUTED)

    # ---- 9 label isolation
    s = D.slide("“Label-free” is enforced in code, not promised", notes=(
        "The central claim is label-free operation, so we made label access impossible rather than discouraged. Layer 1 is a static import-graph test; it is itself mutation-tested against "
        "nine evasion forms. We found that a simpler test, which only reads the module name of an import statement, misses 'from pkg import module'. Layer 2 is a runtime guard. "
        "Layer 3 is behavioural: change the grader's output to random numbers and check that P3/P4/P4c decisions do not change while the oracle's do."))
    ls = [("Static import-graph test", "No observer, detector, decision, classifier or replay module can reach the label store, the labeled loader or the harness, directly or transitively, in any import form. The checker is itself tested against 9 evasion patterns."),
          ("Runtime guard", "The label store checks who is calling and keeps a clock that only moves forward: labels for a week that has not happened yet are refused."),
          ("Behavioural test", "Replace the grader's output with random numbers: the retrain decisions of P3, P4 and P4c stay identical, while the oracle's change. So these decisions do not depend on the grader's labels.")]
    for i, (h, b) in enumerate(ls):
        x = 0.6 + i * 4.1
        D.card(s, x, 1.75, 3.93, 3.7, fill=LIGHT)
        D.box(s, x + 0.2, 1.95, 0.6, 0.6, str(i + 1), fill=NAVY, size=22, radius=0.5)
        D.text(s, x + 0.9, 1.95, 2.95, 0.65, h, size=16, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        D.text(s, x + 0.2, 2.8, 3.55, 2.65, b, size=15)
    D.card(s, 0.6, 5.75, 12.13, 0.95, fill=AMBER_T, accent=AMBER)
    D.text(s, 0.85, 5.78, 11.8, 0.9, "**Found along the way:** a test that only reads the module name of an import statement misses the form “from pkg import module”, so it would pass code that imports the label store. Ours also checks the imported names.",
           size=13.5, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 10 novelty
    s = D.slide("What is new: the combination, judged against the optimum", notes=(
        "This is a positioning, based on how each line of work is described in our literature review; it is not a head-to-head benchmark, because no prior system is label-free, cost-aware "
        "and evaluated on encrypted-traffic classification, so there is nothing identical to run. 'Partly' means the work applies to traffic data but not specifically to encrypted-traffic classifiers."))
    Y, Nn, P_, NA = "Yes", "No", "Partly", "n/a"
    hdr = ["Line of work", "Encrypted traffic", "Decides when to retrain", "No labels to trigger", "Prices retrain cost"]
    mat = [hdr,
           [f"Drift studies in encrypted traffic {c('luxemburk2023', 'kurapov2024', 'jancicka2024', 'malekghaini2023')}", Y, Nn, NA, Nn],
           [f"Label-free drift detectors {c('yang2022', 'lughofer2016', 'sethi2017', 'souza2020')}", P_, Nn, Y, Nn],
           [f"Adaptive retraining triggers {c('gudepu2023', 'bagui2025')}", Nn, Y, Y, Nn],
           [f"Cost-optimal retraining {c('kanda2023')}", Nn, Y, Nn, Y],
           [f"Updating the model once decided {c('liu2023oal', 'cai2024')}", P_, Nn, Nn, Nn],
           ["**This work**", Y, Y, Y, Y]]
    tint = {"Yes": GREEN_T, "No": ORANGE_T, "Partly": AMBER_T, "n/a": LIGHT}
    fills = {}
    for i in range(1, len(mat)):
        for j in range(1, 5):
            fills[(i, j)] = tint[mat[i][j]] if i < len(mat) - 1 else GREEN_T
    for j in range(0, 5):
        fills[(len(mat) - 1, j)] = GREEN_T
    D.table(s, 0.6, 1.7, 7.9, [2.9, 1.2, 1.3, 1.3, 1.2], mat, size=11.5, row_h=0.62, head_size=10.5, fills=fills,
            aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    D.text(s, 0.6, 6.15, 7.9, 0.7, "Positioning of each line of work as summarised in our literature review; not a head-to-head benchmark. “Partly” = traffic data, but not specifically encrypted-traffic classifiers.",
           size=10.5, color=MUTED)
    D.text(s, 8.8, 1.7, 3.93, 0.4, "Contributions", size=17, bold=True, color=NAVY)
    cons = ["**Observer:** 8 label-free signals → one score, oriented and re-baselined per model.",
            "**Decision layer:** frozen loss map plus a cumulative-cost rule for persistent, moderate loss.",
            "**Exact benchmark:** a dynamic-programming bound on what any schedule could achieve.",
            "**Enforced label isolation:** three independent layers.",
            "**Full-year, 5-seed evaluation** plus a QUIC check, negative results included."]
    D.text(s, 8.8, 2.15, 3.93, 4.6, [{"t": t, "bullet": True, "space": 9} for t in cons], size=13.5)

    # ---- 11 setup
    s = D.slide("Experimental setup", notes=(
        "The classifier is a 100-round LightGBM; with library defaults 50-class training diverged on the reference window (accuracy 0.16 with 0.9999 mean max-probability), so we use a hessian floor and L2 regularisation chosen on that window only. "
        "Policies are stand-ins that implement each approach's decision logic on our data; they are not reproductions of the cited systems. The DP frontier tabulates the F1 of every (model, week) pair, 1,275 pairs per seed, and finds the best schedule for each retrain budget."))
    D.text(s, 0.6, 1.65, 4.0, 0.4, "Data and model", size=16, bold=True, color=NAVY)
    D.text(s, 0.6, 2.08, 4.0, 4.7, [{"t": t, "bullet": True, "space": 7} for t in [
        f"**CESNET-TLS-Year22** {c('hynek2024')}: 357 days of 2022, 50 services, 60 flow features; replay weeks 3–52.",
        f"**CESNET-QUIC22** {c('luxemburk2023dib')}: 28 days, decided daily; one documented certificate change.",
        f"**LightGBM** {c('ke2017')} 50-class, held-out macro-F1 **{M['pzeroref']}** at deployment.",
        "**5 seeds**; retrain = fit on the last 2 weeks of labels.",
        "**Hindsight optimum** by dynamic programming over every (model, week) pair."]], size=13.5)
    D.text(s, 4.9, 1.65, 7.8, 0.4, "Policies compared (stand-ins for the literature's approaches)", size=16, bold=True, color=NAVY)
    pol = [["Policy", "Rule", "Stands in for"],
           ["P0 Never", "never retrain", "no maintenance"],
           [f"P1 / P2 Periodic", "every 4 / every 1 weeks", f"calendar retraining {c('kurapov2024')}"],
           ["P3 Threshold", "alarm when D passes a fixed level", f"fixed-threshold baseline {c('gudepu2023')}"],
           ["P4 Myopic cost", "one week's loss vs. cost of fixing", f"cost-based trigger {c('kanda2023')}, but label-free"],
           ["**P4c Cumulative cost**", "**running excess loss vs. cost of fixing**", "**ours**"],
           ["P5 Oracle", "retrain when true loss > 0.05", "label-reading trigger (upper bound)"]]
    D.table(s, 4.9, 2.1, 7.83, [1.7, 3.0, 3.1], pol, size=11.5, row_h=0.46, hl_rows=(5,))
    D.card(s, 4.9, 5.55, 7.83, 1.15, fill=LIGHT)
    D.text(s, 5.1, 5.6, 7.5, 1.05, ["**Metrics:** mean weekly macro-F1, retrains, cost (units of one retrain, weeks above the reference earn no credit), detection delay, false-trigger rate, and **gap to the DP optimum** at the same number of retrains."],
           size=12.5)

    # ---- 12 result 1
    s = D.slide("Result 1: the label-free score tracks the true loss", notes=(
        f"Spearman {num(ck['weekly_level']['spearman'], 2)} and Pearson {num(ck['weekly_level']['pearson'], 2)} between weekly D and the true macro-F1 loss of the never-retrained model over 50 weeks; "
        f"{num(ck['weekly_detrended']['spearman'], 2)} after removing the linear trend from both series; {num(ck['weekly_wow']['spearman'], 2)} week-over-week. The gate (>= 0.3) was fixed before building the cost layer. "
        f"An 11-week check run earlier gave 0.82; the full-year value is lower because D stays high through the second half of the year. The confidence family and reject rate carry the correlation; "
        f"JS divergence of predicted classes is nearly uninformative ({num(sig['js_div']['spearman'], 2)}). Note the composite correlates less than the reject rate alone."))
    D.pic(s, os.path.join(FIGS, "tls.png"), 0.6, 1.65, w=6.9, alt="Top: weekly macro-F1 of never retraining, weekly retraining and the cumulative rule; bottom: drift score D rising after the week-10 exporter update")
    stats = [(num(ck['weekly_level']['spearman'], 2), "Spearman, weekly D vs. true loss"), (num(ck['weekly_detrended']['spearman'], 2), "after removing the trend"),
             (num(ck['weekly_wow']['spearman'], 2), "week-over-week changes")]
    for i, (v, l) in enumerate(stats):
        x = 7.8 + i * 1.65
        D.card(s, x, 1.65, 1.55, 1.3, fill=BLUE_T)
        D.text(s, x, 1.68, 1.55, 0.55, v, size=24, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        D.text(s, x + 0.04, 2.2, 1.47, 0.75, l, size=10.5, align=PP_ALIGN.CENTER, line=1.0)
    order = [("reject_rate", "Reject rate", NAVY), ("margin_mean", "Top-2 margin", BLUE), ("max_prob_p10", "Max prob. (10th pct.)", BLUE),
             ("max_prob_mean", "Max prob. (mean)", BLUE), ("entropy_mean", "Entropy", BLUE), ("ks_mean", "KS statistic", GREEN),
             ("mmd", "MMD²", GREEN), ("js_div", "JS divergence", ORANGE)]
    order[0] = ("reject_rate", "Reject rate", PINK)
    cd = CategoryChartData(); cd.categories = [o[1] for o in order] + ["Composite D"]
    cd.add_series("Spearman", [round(sig[o[0]]["spearman"], 2) for o in order] + [round(ck["weekly_level"]["spearman"], 2)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(7.8), Inches(3.35), Inches(4.93), Inches(3.4), cd)
    ch = gf.chart; ch.has_legend = False; ch.has_title = False; ch.font.size = Pt(11); ch.font.name = FONT
    pl = ch.plots[0]; pl.gap_width = 35; pl.has_data_labels = True
    pl.data_labels.font.size = Pt(11); pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, o in enumerate(order + [("", "", NAVY)]):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(o[2] if i < len(order) else NAVY)
    ch.category_axis.reverse_order = True; ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False
    ch.value_axis.maximum_scale = 1.0; ch.value_axis.minimum_scale = 0
    D.text(s, 7.8, 3.0, 4.93, 0.35, "Spearman with true loss, per signal", size=13, bold=True, color=NAVY)
    D.text(s, 0.6, 6.25, 6.9, 0.65, "Seed 0. Top panel: weekly macro-F1; triangles show when our rule retrained. Bottom: drift score of the never-retrained model.", size=11, color=MUTED)

    # ---- 13 result 2
    s = D.slide("Result 2: policy comparison on a full year of TLS traffic", notes=(
        f"Mean over five seeds at the default cost ratio r=10; macro-F1 is the mean weekly value over 50 replay weeks. Gap is the F1 below the dynamic-programming optimum with the same number of retrains. "
        f"Cost is in units of one retrain, with weeks above the reference model clipped to zero error cost. P4c vs P4: +{d_f1:.3f} F1 with {d_ret:.1f} more retrains; regret to the cost-optimal schedule at r=10 is "
        f"{M['regPfourc']} vs {M['regPfour']} retrain units. Weekly retraining has the best F1 but {M['retrPtwo']} retrains and a false-trigger rate of {M['ftPtwo']}: most of its retrains were not needed."))
    keys = [("P1 monthly (k=4)", "one"), ("P2 weekly (k=1)", "two"), ("Periodic k=8", "eight"), ("P3 threshold alarm", "three"),
            ("P4 myopic cost rule", "four"), ("**P4c cumulative (ours)**", "fourc"), ("P5 threshold oracle", "five")]
    tab2 = [["Policy", "Retrains", "Macro-F1", "Gap to optimum", "Cost", "Delay (weeks)", "False triggers"],
            ["P0 never", "0", M["fonePzero"], "–", M["costPzero"], "–", "–"]]
    for lab, k in keys:
        tab2.append([lab, M[f"retrP{k}"], M[f"foneP{k}"], M[f"gapP{k}"] + " pts", M[f"costP{k}"], M[f"delayP{k}"], M[f"ftP{k}"]])
    hl = [i for i, r in enumerate(tab2) if "ours" in r[0]]
    D.table(s, 0.6, 1.7, 12.13, [3.1, 1.2, 1.3, 1.9, 1.1, 1.6, 1.6], tab2, size=13, row_h=0.36, hl_rows=tuple(hl), head_size=12,
            aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6)
    take = [("P4c vs. the myopic rule", f"**+{d_f1:.3f} F1** for {d_ret:.1f} more retrains; regret to the cost-optimal schedule **{M['regPfourc']} vs {M['regPfour']}** retrain units (r = 10).", GREEN),
            ("Weekly retraining", f"Best F1 ({M['fonePtwo']}) but **{M['retrPtwo']} retrains**; a false-trigger rate of {M['ftPtwo']} means most were not needed.", BLUE),
            ("Fixed threshold alarm", f"No false triggers, but **{M['delayPthree']} weeks late** on average, and it never fired on QUIC.", ORANGE)]
    for i, (h, b, col) in enumerate(take):
        x = 0.6 + i * 4.1
        D.card(s, x, 5.42, 3.93, 1.45, fill=LIGHT, accent=col)
        D.text(s, x + 0.2, 5.45, 3.65, 0.35, h, size=14, bold=True, color=NAVY)
        D.text(s, x + 0.2, 5.82, 3.65, 1.05, b, size=12.5)
    D.text(s, 0.6, 5.02, 12.13, 0.3, "Mean over 5 seeds, r = 10. Cost in units of one retrain; weeks above the reference earn no credit. Gap = F1 below the best possible schedule with the same number of retrains.", size=10, color=MUTED)

    # ---- 14 result 3
    s = D.slide("Result 3: how close to the best possible schedule?", notes=(
        "The dashed line is the highest mean F1 any schedule could reach with a given number of retrains, found by dynamic programming with full hindsight. "
        f"P4c sits {M['gapPfourc']} points below it at its own budget; P4 {M['gapPfour']}; P3 {M['gapPthree']}; the threshold oracle P5 {M['gapPfive']}. At matched budgets a calendar tuned in hindsight is just as close "
        f"(8-week {M['gapPeight']}, monthly {M['gapPone']}), but its interval was chosen using the evaluation year. In cost regret at r=10, P4c is {M['regPfourc']} retrain units above the cost-optimal schedule versus "
        f"{M['regPeriodic']} for the best calendar; P4c has the lowest regret only for r from {M['regRangeLo']} to {M['regRangeHi']}: at r=300 the tuned calendar does better ({M['regPeriodicHigh']} vs {M['regPfourcHigh']}). "
        "Weekly retraining (49 retrains, F1 0.868) is off the right edge of the chart."))
    D.pic(s, os.path.join(FIGS, "pareto.png"), 0.6, 1.65, w=6.8, alt="Mean macro-F1 versus number of retrains for each policy family with the hindsight-optimal frontier")
    D.text(s, 0.6, 6.2, 6.8, 0.6, "Mean of 5 seeds. Curves for P3, P4, P4c and P5 are traced by sweeping their knobs. Weekly retraining (49 retrains) is off the right edge.", size=10.5, color=MUTED)
    pts = [f"**Dashed line:** the best mean F1 any schedule could reach with that many retrains, found with full hindsight.",
           f"**P4c is {M['gapPfourc']} pts below it** at its own budget; P4 {M['gapPfour']}, threshold alarm {M['gapPthree']}, threshold oracle {M['gapPfive']}.",
           f"**At matched budgets, tuned calendars are just as close** (8-week {M['gapPeight']}, monthly {M['gapPone']}). Their interval was chosen on the evaluation year; ours was not.",
           f"**Cost regret at r = 10:** P4c {M['regPfourc']} vs {M['regPeriodic']} for the best calendar. P4c is lowest only for r from {M['regRangeLo']} to {M['regRangeHi']}; at r = 300 the tuned calendar wins ({M['regPeriodicHigh']} vs {M['regPfourcHigh']})."]
    D.text(s, 7.7, 1.7, 5.03, 5.1, [{"t": t, "bullet": True, "space": 12} for t in pts], size=14.5)

    # ---- 15 result 4
    q = N["quic"]
    seeds_q = sorted(q["seeds"], key=int)
    s = D.slide("Result 4: on QUIC it retrains on the day of the certificate change", notes=(
        "The spec asked us to confirm that the trigger fires at the documented certificate change. In all five seeds P4 and P4c retrain on day 22 with no earlier retrain, so zero delay; the threshold alarm never fires. "
        f"The picture is less clean than one step: the oracle retrains earlier (days 9-18) because the model had already lost accuracy before the change, and a change point appears in D at day 11 or 12. "
        f"The plain before/after F1 gap is small and noisy (mean {M['qdrop']}), so we do not offer it as evidence; the timing match is. The five seeds share one physical event, so 5/5 measures stability, not five confirmations. "
        "Early in this work we found a configuration-merge bug that silently used the TLS event day (10) here; it is fixed and covered by a test."))
    D.pic(s, os.path.join(FIGS, "quic.png"), 0.6, 1.65, w=6.3, alt="QUIC22 daily macro-F1 for never retraining, our rule and the oracle, with the day-22 certificate change marked")
    def days(v):
        return "never" if not v else ", ".join(str(x) for x in v)
    qt = [["Retrain day(s)"] + [f"S{sd}" for sd in seeds_q]]
    for lab, key in (("Threshold alarm (P3)", "P3_threshold"), ("Myopic rule (P4)", "P4_proposed"), ("**Cumulative (P4c)**", "P4c_cumulative"), ("Oracle (P5)", "P5_oracle")):
        qt.append([lab] + [days(q["seeds"][sd][key]["retrain_days"]) for sd in seeds_q])
    qt.append(["Change point in D"] + [days(q["seeds"][sd]["drift_event_changepoint_days"]) for sd in seeds_q])
    fills = {(i, j): (GREEN_T if qt[i][j] == "22" else (ORANGE_T if qt[i][j] == "never" else (LIGHT if i % 2 == 0 else WHITE))) for i in range(1, len(qt)) for j in range(1, 6)}
    D.table(s, 7.2, 1.7, 5.53, [2.3, 0.9, 0.9, 0.9, 0.9, 0.9], qt, size=11.5, row_h=0.46, fills=fills, aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 5, head_size=11)
    pts = ["**5 of 5 seeds:** P4 and P4c retrain on day 22 with no earlier retrain: zero delay.",
           "**The fixed threshold never fires** in any seed.",
           "**The oracle retrains earlier** (days 9–18): accuracy had already fallen before day 22, and D has a change point at day 11–12.",
           "**Caveat:** the five seeds share one physical event: this shows stability, not five independent confirmations."]
    D.text(s, 7.2, 4.6, 5.53, 2.3, [{"t": t, "bullet": True, "space": 5} for t in pts], size=11.5)
    D.text(s, 0.6, 6.3, 6.3, 0.55, "Seed 0. Triangles: retrains (amber: ours, pink: oracle). Dashed line: documented certificate change.", size=10.5, color=MUTED)

    # ---- 16 comparison 1
    s = D.slide("Compared with the literature (1/2): where our results agree", notes=(
        "Each row is a claim from the surveyed literature, what we measured, and a verdict. These are comparisons of findings, not re-implementations. "
        f"On row 1 the metrics differ (they report accuracy on QUIC; we report macro-F1 with a different model), so the match is in direction and order of magnitude only. "
        f"TLS: macro-F1 {M['pzerowkNine']} in week 9 to {M['pzerowkEleven']} in week 11 is {exp_drop:.1f} points; QUIC seed 0: {M['qfdayeight']} on day 8 to {M['qfdaytwentyone']} on day 21. "
        "Row 5: we observed the false-alarm side of the PELT trade-off; we did not test the missed-detection side."))
    rowsA = [["Claim in the literature", "What we measured", "Verdict"],
             [f"{c('luxemburk2023')} Classifiers lose 9–13 accuracy points within three weeks of a provider change (QUIC).",
              f"TLS macro-F1 fell **{exp_drop:.1f} points** from week 9 to week 11, across the week-10 exporter update; QUIC never-retrained F1 fell {M['qfdayeight']} → {M['qfdaytwentyone']} (day 8 to 21).",
              "Consistent in direction and size (different metric and model)"],
             [f"{c('gudepu2023')} Fixed-threshold and periodic retraining are both insufficient.",
              f"Threshold alarm: **{M['delayPthree']}-week** delay, never fired on QUIC. Weekly calendar: {M['retrPtwo']} retrains, false-trigger rate {M['ftPtwo']}.",
              "Supported (a tuned calendar is still competitive)"],
             [f"{c('kanda2023')} Detector-based policies cost less than periodic ones, but the detector reads labels.",
              f"Label-free P4c: cost regret **{M['regPfourc']}** vs **{M['regPeriodic']}** for the best calendar (r = 10); lowest regret of the compared policies for r = {M['regRangeLo']}–{M['regRangeHi']}.",
              "Supported without labels"],
             [f"{c('lughofer2016', 'sethi2017')} Classifier confidence and margin are usable label-free drift signals.",
              f"Confidence family: Spearman **{num(min(sig[k]['spearman'] for k in ('max_prob_mean', 'max_prob_p10', 'margin_mean', 'entropy_mean')), 2)}–{num(max(sig[k]['spearman'] for k in ('max_prob_mean', 'max_prob_p10', 'margin_mean', 'entropy_mean')), 2)}** with true loss; reject rate {num(sig['reject_rate']['spearman'], 2)}.",
              "Strongly supported"],
             [f"{c('smirnov2023')} The PELT penalty is a real trade-off: too small gives false alarms.",
              f"On calm weeks, penalties 1–5 gave 3–4 false change points; 8 and above gave none (we use {int(CAL0['pelt_penalty'])}).",
              "Observed (false-alarm side only)"]]
    fl = {(i, 2): GREEN_T for i in range(1, 6)}
    D.table(s, 0.6, 1.7, 12.13, [4.0, 5.2, 2.5], rowsA, size=13, row_h=0.92, fills=fl, head_size=13, head_h=0.45)

    # ---- 17 comparison 2
    s = D.slide("Compared with the literature (2/2): where we refine or differ", notes=(
        "Here our results refine, extend, or do not reproduce a claim. Row 3: the cited method compares traffic distributions between windows, while we use Jensen-Shannon divergence on the predicted-class histogram; "
        "so this is not a refutation of that paper, only evidence that this signal is weak in our setting. Row 4: LEAF studies cellular networks; in our closed-world setting with a 2-week window "
        "no retraining policy ended below never retraining on mean F1."))
    rowsB = [["Claim or gap in the literature", "What we measured", "Verdict"],
             [f"{c('kurapov2024')} Monthly retraining suffices; they could not locate when quality starts to degrade.",
              f"Monthly (12 retrains): F1 **{M['fonePone']}**, {M['gapPone']} pts from optimum. P4c: {M['retrPfourc']} retrains, F1 {M['fonePfourc']}, reacts **{M['delayPfourc']} weeks** after loss passes 0.03: the “when” they lacked.",
              "Refines: sufficient, but not frugal"],
             [f"{c('jancicka2024')} There is no agreed level at which a feature-distribution change is “meaningful”.",
              "We price it: g maps the score to F1 loss, and the decision depends on the cost ratio r, not on a fixed level.",
              "Answers the open question (r is operator-chosen)"],
             [f"{c('yang2022')} Jensen–Shannon divergence detects drift without labels.",
              f"Our JS signal (on predicted-class histograms, not traffic windows) has Spearman **{num(sig['js_div']['spearman'], 2)}** with true loss.",
              "Not supported in our setting"],
             [f"{c('liu2023leaf')} Retraining can make accuracy worse (cellular networks).",
              "No retraining policy ended below never-retraining on mean F1 (closed world, 2-week retrain window).",
              "Not observed here"],
             [f"{c('bagui2025')} Detect-then-retrain loops keep models in sync (network-security data).",
              "Same loop, but label-free, cost-priced, on encrypted-traffic classification, and judged against the hindsight optimum.",
              "Extends"]]
    fl = {(1, 2): BLUE_T, (2, 2): BLUE_T, (3, 2): ORANGE_T, (4, 2): AMBER_T, (5, 2): BLUE_T}
    D.table(s, 0.6, 1.7, 12.13, [4.0, 5.2, 2.5], rowsB, size=13, row_h=0.92, fills=fl, head_size=13, head_h=0.45)

    # ---- 18 limitations
    gap = float(M["gtruemean"]) - float(M["gpredmean"])
    s = D.slide("Limitations we measured, not guessed", notes=(
        f"Each item is a measured weakness. g: in weeks 40-45 it predicts mean loss {M['gpredmean']} where the true loss of the never-retrained model is {M['gtruemean']}; it ranks days correctly (Spearman {M['gtransrho']}) "
        "but under-predicts the level, which biases both cost rules toward waiting and is consistent with P4c losing to a tuned calendar at r=300. "
        f"The composite D has Spearman {num(ck['weekly_level']['spearman'], 2)} against {num(sig['reject_rate']['spearman'], 2)} for the reject rate alone. We did not run an ablation of signals inside the decision rules."))
    lim = [("Labels are rationed, not eliminated", "The loss map needs 8 labeled weeks, and each retrain needs 2.", ORANGE),
           ("The loss map ages", f"In weeks 40–45 it predicts mean loss **{M['gpredmean']}** where the truth is **{M['gtruemean']}** (ranking still good, Spearman {M['gtransrho']}). Both cost rules then wait too long; P4c loses to a tuned calendar at r = 300.", ORANGE),
           ("Not every signal pulls its weight", f"The composite (Spearman {num(ck['weekly_level']['spearman'], 2)}) is worse than the reject rate alone ({num(sig['reject_rate']['spearman'], 2)}). JS, KS and MMD add little here; no signal ablation in the decision rules.", AMBER),
           ("A tuned calendar is a near-tie", "At matched retrain budgets, a hindsight-tuned periodic schedule is as close to the optimum. Our edge: timing that follows the traffic, no tuning, and a cost knob.", AMBER),
           ("QUIC evidence is thin", "28 days and one documented event; the five seeds share it.", BLUE),
           ("Scope", "Illustrative cost constants, a 50-service closed world, sampled flows from one vantage point.", BLUE)]
    for i, (h, b, col) in enumerate(lim):
        x, y = 0.6 + (i % 2) * 6.15, 1.7 + (i // 2) * 1.7
        D.card(s, x, y, 5.98, 1.58, fill=LIGHT, accent=col)
        D.text(s, x + 0.2, y + 0.03, 5.65, 0.4, h, size=14.5, bold=True, color=NAVY)
        D.text(s, x + 0.2, y + 0.43, 5.65, 1.12, b, size=13.5)

    # ---- 19 conclusion
    s = D.slide("Conclusion", notes=(
        "Four findings, then what we would do next. The code, the label-isolation tests and the exact-optimum harness are in the repository so the comparisons can be repeated on other traffic."))
    cs = [(f"{M['pzeroref']} → {M['pzeromin']}", "macro-F1 of a never-retrained classifier over 2022: drift is real.", ORANGE),
          (f"{num(ck['weekly_level']['spearman'], 2)} / {num(ck['weekly_detrended']['spearman'], 2)}", "Spearman of the label-free score with true loss (raw / detrended).", BLUE),
          (f"{M['fonePfourc']} F1 at {M['retrPfourc']} retrains", f"cumulative rule: {M['gapPfourc']} pts from the hindsight optimum; beats the myopic rule ({M['fonePfour']}).", AMBER),
          ("5 of 5 seeds", "retrain on the day of the QUIC certificate change; the fixed threshold never fires.", GREEN)]
    for i, (v, b, col) in enumerate(cs):
        y = 1.7 + i * 1.28
        D.card(s, 0.6, y, 7.5, 1.15, fill=LIGHT, accent=col)
        D.text(s, 0.8, y + 0.05, 3.1, 1.05, v, size=20, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        D.text(s, 3.95, y + 0.05, 4.1, 1.05, b, size=13, anchor=MSO_ANCHOR.MIDDLE)
    D.text(s, 8.5, 1.7, 4.23, 0.4, "Next steps", size=17, bold=True, color=NAVY)
    D.text(s, 8.5, 2.15, 4.23, 3.0, [{"t": t, "bullet": True, "space": 8} for t in [
        "Recalibrate the loss map g online.", "Ablate signals inside the decision rules.", "Partial retraining instead of full refits.", "Open-world traffic and more events."]], size=14)
    D.card(s, 8.5, 5.3, 4.23, 1.5, fill=NAVY)
    D.text(s, 8.7, 5.35, 3.9, 1.4, ["**Code and tests**", "github.com/Darsh2905/Label-free-retraining-triage"], size=13, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 20 references
    s = D.slide("References", notes="Numbering follows the paper's bibliography. Entries were checked against Crossref.")
    bib = open(os.path.join(PAPER, "refs.bib")).read()
    def field(entry, name):
        m = re.search(name + r"=\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", entry)
        return m.group(1) if m else ""
    def clean(t):
        for a, b in {"{\\v{C}}": "Č", "{\\v{c}}": "č", "{\\v{S}}": "Š", "{\\v{s}}": "š", "{\\v{Z}}": "Ž", "{\\.{e}}": "ė", '{\\"o}': "ö"}.items():
            t = t.replace(a, b)
        return re.sub(r"[{}]", "", t).replace("--", "–")
    entries = {m.group(2): m.group(0) for m in re.finditer(r"@(\w+)\{(\w+),.*?(?=\n@|\Z)", bib, re.S)}
    items = []
    for k in sorted(USED, key=NUM.get):
        e = entries[k]
        au = clean(field(e, "author")).split(" and ")
        first = au[0].split(". ")[-1] if au else ""
        who = f"{au[0]}" if len(au) == 1 else f"{au[0]} et al."
        venue = clean(field(e, "journal") or field(e, "booktitle"))
        venue = re.sub(r"^Proc\. ", "", venue)
        items.append(f"[{NUM[k]}] {who}, “{clean(field(e, 'title'))},” {venue}, {field(e, 'year')}.")
    half = (len(items) + 1) // 2
    for col, chunk in enumerate((items[:half], items[half:])):
        D.text(s, 0.6 + col * 6.15, 1.65, 5.98, 5.25, [{"t": t, "space": 6} for t in chunk], size=11, line=1.0)
    D.prs.save(path)
    return D


if __name__ == "__main__":
    out = os.path.join(DECK, "triage_deck.pptx")
    d = build(out)
    print("saved", out, "slides:", d.n)
    if WARN:
        print("LAYOUT WARNINGS:")
        for w in WARN:
            print("  -", w)
    else:
        print("no layout warnings")
