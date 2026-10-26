"""paper/style.py -- one visual system for every GEMMA paper figure (ACM sigconf).

Sizes are the FINAL printed sizes, so fonts land at 6.5-8 pt:
  COL  = 3.33 in  (one column, \\columnwidth)      TEXT = 7.0 in (\\textwidth, figure*)
Fonts: Libertinus Sans (= ACM's Biolinum look) from paper/fonts, falling back to DejaVu Sans.
Colours: roles, not series numbers -- validated with the dataviz CVD checker:
  OURS  #EB6834  (GEMMA, every figure)          LLM   #2A78D6  (LLM agents)
  RET   #148A60  (return channel)               EXP   #6250D6  (explore channel)
  SOTA  #52514E  (STHGCN = strongest supervised) BASE  #B4B2AB  (other supervised baselines)
Zero-shot vs fine-tuned GEMMA is told apart by FILL (hollow vs solid) as well as tint, never by colour alone.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "figs")

COL, TEXT = 3.33, 7.0
OURS, OURS_LIGHT = "#EB6834", "#F6B79C"
LLM, LLM_LIGHT = "#2A78D6", "#A9C9F0"
RET, EXP = "#148A60", "#6250D6"
SOTA, BASE, GRID = "#52514E", "#B4B2AB", "#E4E2DD"
INK, INK2, MUTED = "#1F1F1D", "#52514E", "#8A8984"
SIG = "#1F1F1D"

SRC = ["istanbul", "petaling_jaya", "jakarta", "new_york", "kuwait_city", "sao_paulo"]
# Beijing is excluded from every evaluation and figure of the paper (88 test queries: too few for a stable estimate).
EXCLUDED = {"beijing"}
TGT = ["tokyo", "shanghai", "palembang", "melbourne", "sydney", "tangerang", "moscow", "bandung"]
N_TGT = len(TGT)
PRETTY = {"istanbul": "Istanbul", "petaling_jaya": "Petaling Jaya", "jakarta": "Jakarta", "new_york": "New York",
          "kuwait_city": "Kuwait City", "sao_paulo": "São Paulo", "tokyo": "Tokyo", "beijing": "Beijing",
          "shanghai": "Shanghai", "palembang": "Palembang", "melbourne": "Melbourne", "sydney": "Sydney",
          "tangerang": "Tangerang", "moscow": "Moscow", "bandung": "Bandung"}
BY_PRETTY = {v: k for k, v in PRETTY.items()}


def pretty(c):
    return PRETTY.get(c, c)


def setup():
    fam = "DejaVu Sans"
    fdir = os.path.join(HERE, "fonts")
    if os.path.isdir(fdir):
        for f in os.listdir(fdir):
            if f.endswith(".ttf"):        # TrueType copies (made from the OFL OTFs) embed cleanly as Type 42
                font_manager.fontManager.addfont(os.path.join(fdir, f))
        if any("Libertinus Sans" in f.name for f in font_manager.fontManager.ttflist):
            fam = "Libertinus Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [fam, "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.title_fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": .2,
        "lines.linewidth": 1.4, "lines.markersize": 4, "patch.linewidth": 0,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 4,
    })
    os.makedirs(OUT, exist_ok=True)


def panel(ax, letter, title=None, x=-0.02, y=1.0):
    """'(a) Title' bold, left aligned above the axes -- the ACM/WWW convention."""
    ax.set_title(f"({letter}) {title}" if title else f"({letter})", loc="left", x=x, y=y)


def grid_y(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("[figure]", os.path.join(OUT, name + ".pdf"))
