"""Generates high-DPI benchmark charts and the 16-slide doctor-ready Graduation Defense PowerPoint."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT_PPTX = ROOT / "docs" / "AI_Personal_Gym_Companion_Final_Defense.pptx"
ASSETS_DIR = Path(
    r"C:\Users\riadk\.gemini\antigravity\brain\dd804d97-60b5-4387-bd3d-b2732aa1bdf8\scratch\pptx_assets"
)
OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Executive Dark Navy Palette
BG_DARK = RGBColor(11, 17, 32)       # #0B1120
CARD_BG = RGBColor(19, 28, 49)       # #131C31
CARD_ALT = RGBColor(23, 35, 61)      # #17233D
BORDER = RGBColor(38, 53, 84)        # #263554
EMERALD = RGBColor(16, 185, 129)     # #10B981
CYAN = RGBColor(56, 189, 248)        # #38BDF8
AMBER = RGBColor(245, 158, 11)       # #F59E0B
RED_ERR = RGBColor(248, 113, 113)    # #F87171
TEXT_MAIN = RGBColor(248, 250, 252)  # #F8FAFC
TEXT_BODY = RGBColor(203, 213, 225)  # #CBD5E1
TEXT_MUTED = RGBColor(148, 163, 184) # #94A3B8
DARK_TXT = RGBColor(6, 20, 16)

FONT_FAMILY = "Segoe UI"
TOTAL_SLIDES = 16


def _family_hex(fam: str) -> str:
    low = fam.lower()
    if "merged" in low:
        return "#F59E0B"
    if "deep" in low:
        return "#38BDF8"
    return "#10B981"


def generate_leaderboard_chart(csv_path: Path, title: str, highlight_model: str, out_png: Path) -> None:
    df = pd.read_csv(csv_path).sort_values("OOF Macro F1", ascending=True).reset_index(drop=True)
    plt.rcParams.update({"font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=300)
    fig.patch.set_facecolor("#131C31")
    ax.set_facecolor("#0F172A")

    colors = [_family_hex(f) for f in df["Family"]]
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["OOF Macro F1"], color=colors, height=0.62, edgecolor="#1E293B", linewidth=0.8)

    has_lat = "Latency (ms/rep)" in df.columns
    for idx, bar in enumerate(bars):
        row = df.iloc[idx]
        is_top = idx == len(df) - 1
        if is_top:
            bar.set_edgecolor("#FFFFFF")
            bar.set_linewidth(1.5)
        f1_v = float(row["OOF Macro F1"])
        auc_v = float(row["ROC-AUC"])
        if has_lat and pd.notna(row["Latency (ms/rep)"]):
            lbl = f" F1={f1_v:.4f} | AUC={auc_v:.3f} | {float(row['Latency (ms/rep)']):.1f}ms"
        else:
            acc_v = float(row["Accuracy"])
            lbl = f" F1={f1_v:.4f} | AUC={auc_v:.4f} | Acc={acc_v:.1%}"
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2.0,
            lbl,
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold" if is_top else "normal",
            color="#FFFFFF" if is_top else "#CBD5E1",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Model"], fontsize=9, color="#E2E8F0", fontweight="bold")
    ax.set_xlim(0.0, min(1.30, float(df["OOF Macro F1"].max()) + 0.36))
    ax.set_xlabel("Out-of-Fold Macro F1 Score (5-Fold Subject-Grouped CV)", fontsize=9.5, color="#94A3B8", labelpad=6)
    ax.set_title(title, fontsize=11.2, fontweight="bold", color="#F8FAFC", pad=10, loc="left")
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=8.5)
    ax.grid(axis="x", color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#334155")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#10B981", label="Classical / Tabular (178 Phase Features)"),
        Patch(facecolor="#38BDF8", label="Deep Learning (1D / 2D Skeleton / RNN / TCN)"),
        Patch(facecolor="#F59E0B", label="Merged Hybrid & Cross-Family Ensemble"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#0B1120", edgecolor="#334155", fontsize=7.6, labelcolor="#E2E8F0")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_all6_rf_chart(out_png: Path) -> None:
    """Plots OOF Macro F1 and ROC-AUC across all 6 deployed production models."""
    ex_specs = [
        ("shoulder", "Shoulder Press (RF)\n(25 subj · 1,961 reps)"),
        ("push", "Push-Up (RF)\n(35 subj · Multi-View)"),
        ("lunge", "Lunge (Ensemble)\n(UI-PRMD · 255 reps)"),
        ("arm_abduction", "Arm Abduction (Ensemble)\n(UI-PRMD · 144 reps)"),
        ("squat", "Squat (Ensemble)\n(UI-PRMD · 178 reps)"),
        ("curl", "Bicep Curl (Ensemble)\n(NgoQuocBao · 439 reps)"),
    ]
    labels, f1s, aucs = [], [], []
    for ex_key, disp in ex_specs:
        df = pd.read_csv(ROOT / "backend" / "models" / ex_key / "model_leaderboard.csv")
        ens = df[df["Model"].str.contains("Unified Ensemble", na=False)]
        row = ens.iloc[0] if len(ens) else df[df["Model"] == "Random Forest"].iloc[0]
        labels.append(disp)
        f1s.append(float(row["OOF Macro F1"]))
        aucs.append(float(row["ROC-AUC"]))

    plt.rcParams.update({"font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(7.3, 4.45), dpi=300)
    fig.patch.set_facecolor("#131C31")
    ax.set_facecolor("#0F172A")

    y = np.arange(len(labels))
    h = 0.36
    b1 = ax.barh(y + h / 2, f1s, height=h, color="#10B981", edgecolor="#1E293B", label="OOF Macro F1")
    b2 = ax.barh(y - h / 2, aucs, height=h, color="#38BDF8", edgecolor="#1E293B", label="ROC-AUC")

    for bar, v in zip(b1, f1s):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", color="#A7F3D0", fontsize=8.3, fontweight="bold")
    for bar, v in zip(b2, aucs):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", color="#BAE6FD", fontsize=8.3, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.4, color="#E2E8F0", fontweight="bold")
    ax.set_xlim(0.0, 1.14)
    ax.set_xlabel("Subject-Grouped 5-Fold Cross-Validation Score", fontsize=9.2, color="#94A3B8")
    ax.set_title("Deployed Production Models Across 6 Exercises (Unified Ensemble + RF)", fontsize=10.6, fontweight="bold", color="#F8FAFC", loc="left", pad=10)
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=8.4)
    ax.grid(axis="x", color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#334155")
    ax.legend(loc="lower right", frameon=True, facecolor="#0B1120", edgecolor="#334155", fontsize=8.2, labelcolor="#E2E8F0")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_smartfit_chart(out_png: Path) -> None:
    models = ["MLP Neural Net\n(64, 32)", "SVM (RBF)\n(C=2.0, Balanced)", "Random Forest\n(250 Trees, Selected)"]
    f1s = [0.4521, 0.4555, 0.4589]
    accs = [0.454, 0.457, 0.459]
    colors = ["#38BDF8", "#60A5FA", "#10B981"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)
    fig.patch.set_facecolor("#131C31")
    ax.set_facecolor("#0F172A")

    y = np.arange(len(models))
    bars = ax.barh(y, f1s, color=colors, height=0.52, edgecolor="#1E293B")
    bars[-1].set_edgecolor("#FFFFFF")
    bars[-1].set_linewidth(1.5)

    for b, f1, acc in zip(bars, f1s, accs):
        ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2, f"F1={f1:.4f} | Acc={acc:.1%}", va="center", color="#F8FAFC", fontsize=9, fontweight="bold")

    ax.axvline(0.3333, color="#F59E0B", linestyle="--", linewidth=1.2)
    ax.text(0.336, -0.38, "3-Class Random Chance (0.333)", color="#FBBF24", fontsize=7.8)
    ax.set_yticks(y)
    ax.set_yticklabels(models, color="#E2E8F0", fontsize=9, fontweight="bold")
    ax.set_xlim(0.30, 0.56)
    ax.set_xlabel("5-Fold Stratified CV Macro F1 (n=1,000 SmartFit Athletes)", color="#94A3B8", fontsize=8.8)
    ax.set_title("SmartFit Fitness-Level Predictor Benchmark (Signup AI)", color="#F8FAFC", fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=8.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#334155")
    ax.grid(axis="x", color="#1E293B", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def extract_mid_frame(video_path: Path, ratio: float = 0.50) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 30)
    target = max(1, min(total - 2, int(total * ratio)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return np.zeros((360, 520, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def generate_demo_collage(out_png: Path) -> None:
    curl_candidates = sorted(
        (ROOT / "backend" / "data" / "outputs").glob("annotated_auto_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    curl_mp4 = curl_candidates[0] if curl_candidates else (ROOT / "demo_outputs" / "rohanx01_squat (1)_coach_demo.mp4")

    samples = [
        (ROOT / "demo_outputs" / "male-Barbell-barbell-squat-front (1)_coach_demo.mp4", 0.52, "Squat — Clean Form Verified\n(Random Forest + 3D Pose HUD)"),
        (ROOT / "demo_outputs" / "squat-two-incorrect-reps_coach_demo.mp4", 0.48, "Squat — Fault Caught & Voice Cue\n(\"Keep back straight / Go deeper\")"),
        (curl_mp4, 0.55, "Auto-Detected Bicep Curl (H.264)\n(Independent L/R Arm Tracker)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), dpi=280)
    fig.patch.set_facecolor("#131C31")
    for ax, (vp, rat, caption) in zip(axes, samples):
        img = extract_mid_frame(vp, rat)
        ax.imshow(img)
        ax.set_title(caption, fontsize=9.2, fontweight="bold", color="#F8FAFC", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#38BDF8")
            sp.set_linewidth(1.5)
    plt.tight_layout(w_pad=1.2)
    fig.savefig(out_png, dpi=280, facecolor=fig.get_facecolor())
    plt.close(fig)


# -------- Slide Helpers --------
def set_dark_bg(slide) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK


def add_header(slide, eyebrow: str, title: str, slide_num: int) -> None:
    set_dark_bg(slide)
    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.32), Inches(0.55), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = EMERALD
    bar.line.fill.background()

    eb = slide.shapes.add_textbox(Inches(1.22), Inches(0.20), Inches(8.5), Inches(0.28))
    tf_eb = eb.text_frame
    tf_eb.margin_left = tf_eb.margin_top = 0
    p_eb = tf_eb.paragraphs[0]
    p_eb.text = eyebrow.upper()
    p_eb.font.name = FONT_FAMILY
    p_eb.font.size = Pt(10.5)
    p_eb.font.bold = True
    p_eb.font.color.rgb = EMERALD

    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.44), Inches(12.2), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = 0
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT_FAMILY
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    # Thin divider
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.06), Inches(12.23), Inches(0.015))
    div.fill.solid()
    div.fill.fore_color.rgb = BORDER
    div.line.fill.background()

    # Footer
    ft_l = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(8.5), Inches(0.26))
    p_fl = ft_l.text_frame.paragraphs[0]
    p_fl.text = "AI Personal Gym Companion (FormCoach)  ·  Graduation Defense  ·  Team AI25"
    p_fl.font.name = FONT_FAMILY
    p_fl.font.size = Pt(9.5)
    p_fl.font.color.rgb = TEXT_MUTED

    ft_r = slide.shapes.add_textbox(Inches(11.3), Inches(7.12), Inches(1.48), Inches(0.26))
    p_fr = ft_r.text_frame.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.text = f"{slide_num} / {TOTAL_SLIDES}"
    p_fr.font.name = FONT_FAMILY
    p_fr.font.size = Pt(9.5)
    p_fr.font.bold = True
    p_fr.font.color.rgb = CYAN


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str = "",
    bullets: list[str] | None = None,
    accent: RGBColor = EMERALD,
    fill_color: RGBColor = CARD_BG,
    body_size: float = 12.5,
) -> object:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.04
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    card.line.color.rgb = BORDER
    card.line.width = Pt(1.0)

    # Top accent strip inside card
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left + 0.14), Inches(top + 0.12), Inches(min(0.95, width * 0.28)), Inches(0.045))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.22), Inches(width - 0.36), Inches(height - 0.32))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    first = True
    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(14.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)
        first = False

    if bullets:
        for b in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = b if b.startswith(("•", "▸", "—", "1.", "2.", "3.", "4.", "5.")) else f"▸  {b}"
            p.font.name = FONT_FAMILY
            p.font.size = Pt(body_size)
            p.font.color.rgb = TEXT_BODY
            p.space_after = Pt(4.5)
    return card


def add_kpi_tile(slide, left: float, top: float, width: float, height: float, value: str, label: str, accent: RGBColor = EMERALD) -> None:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_ALT
    box.line.color.rgb = accent
    box.line.width = Pt(1.25)

    tb = slide.shapes.add_textbox(Inches(left + 0.12), Inches(top + 0.10), Inches(width - 0.24), Inches(height - 0.18))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.text = value
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(21)
    p1.font.bold = True
    p1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = label
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_BODY


def style_table(table, col_widths: list[float], headers: list[str], rows: list[list[str]], font_size: float = 11.0) -> None:
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = Inches(w)

    for c_i, h_txt in enumerate(headers):
        cell = table.cell(0, c_i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_ALT
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_txt
        p.font.name = FONT_FAMILY
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = EMERALD

    for r_i, row_vals in enumerate(rows, 1):
        bg_c = CARD_BG if r_i % 2 == 1 else RGBColor(16, 24, 42)
        for c_i, val in enumerate(row_vals):
            cell = table.cell(r_i, c_i)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_FAMILY
            p.font.size = Pt(font_size - 0.5)
            p.font.bold = (c_i == 0)
            p.font.color.rgb = TEXT_MAIN if c_i == 0 else TEXT_BODY


def build_deck() -> None:
    squat_csv = ROOT / "backend" / "models" / "squat" / "model_leaderboard.csv"
    push_csv = ROOT / "backend" / "models" / "push" / "model_leaderboard.csv"

    squat_png = ASSETS_DIR / "squat_leaderboard.png"
    push_png = ASSETS_DIR / "push_leaderboard.png"
    all6_rf_png = ASSETS_DIR / "all6_rf_summary.png"
    smartfit_png = ASSETS_DIR / "smartfit_cv.png"
    collage_png = ASSETS_DIR / "demo_frames_collage.png"

    generate_leaderboard_chart(squat_csv, "Squat (UI-PRMD) — Subject-Grouped OOF Benchmark (232 Reps)", "Random Forest", squat_png)
    generate_leaderboard_chart(push_csv, "Push-Up — 10-Model OOF Benchmark (Mendeley + REHAB24-6)", "Ensemble", push_png)
    generate_all6_rf_chart(all6_rf_png)
    generate_smartfit_chart(smartfit_png)
    generate_demo_collage(collage_png)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # === SLIDE 1: TITLE SLIDE ===
    s1 = prs.slides.add_slide(blank)
    set_dark_bg(s1)
    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.22), Inches(7.5))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = EMERALD
    accent_bar.line.fill.background()

    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.15), Inches(4.5), Inches(0.42))
    badge.adjustments[0] = 0.5
    badge.fill.solid()
    badge.fill.fore_color.rgb = CARD_ALT
    badge.line.color.rgb = EMERALD
    p_b = badge.text_frame.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    p_b.text = "GRADUATION PROJECT DEFENSE  ·  TEAM AI25"
    p_b.font.name = FONT_FAMILY
    p_b.font.size = Pt(11.5)
    p_b.font.bold = True
    p_b.font.color.rgb = EMERALD

    tb_t = s1.shapes.add_textbox(Inches(0.85), Inches(1.85), Inches(11.5), Inches(1.6))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    pt1 = tf_t.paragraphs[0]
    pt1.text = "AI Personal Gym Companion"
    pt1.font.name = FONT_FAMILY
    pt1.font.size = Pt(46)
    pt1.font.bold = True
    pt1.font.color.rgb = TEXT_MAIN
    pt2 = tf_t.add_paragraph()
    pt2.text = "FormCoach: Real-Time Computer Vision, Biometric HUD & Multi-Model Form Assessment"
    pt2.font.name = FONT_FAMILY
    pt2.font.size = Pt(20)
    pt2.font.color.rgb = CYAN

    add_kpi_tile(s1, 0.85, 4.05, 2.75, 1.15, "7 + Auto", "6/7 ML + Stage-1 AI", EMERALD)
    add_kpi_tile(s1, 3.85, 4.05, 2.75, 1.15, "0.977 / 0.927", "Curl & Squat Ensemble F1", CYAN)
    add_kpi_tile(s1, 6.85, 4.05, 2.75, 1.15, "3-Angle ID", "YuNet + SFace Biometric HUD", AMBER)
    add_kpi_tile(s1, 9.85, 4.05, 2.75, 1.15, "Web + Flutter", "FastAPI WS & TTS Coach", EMERALD)

    tm = s1.shapes.add_textbox(Inches(0.85), Inches(5.85), Inches(11.5), Inches(0.95))
    tf_tm = tm.text_frame
    ptm1 = tf_tm.paragraphs[0]
    ptm1.text = "Team Members:  Mohamed Shawky  ·  Ahmed El Kholy  ·  Fouad Khaled"
    ptm1.font.name = FONT_FAMILY
    ptm1.font.size = Pt(15.5)
    ptm1.font.bold = True
    ptm1.font.color.rgb = TEXT_MAIN
    ptm2 = tf_tm.add_paragraph()
    ptm2.text = "Department of Artificial Intelligence  |  November 2026  |  Live Backend + Flutter + Multi-View Colab Pipeline"
    ptm2.font.name = FONT_FAMILY
    ptm2.font.size = Pt(12)
    ptm2.font.color.rgb = TEXT_MUTED

    # === SLIDE 2: PROBLEM STATEMENT & MOTIVATION ===
    s2 = prs.slides.add_slide(blank)
    add_header(s2, "01 · Problem Statement", "Why Workout Apps Need Real-Time Computer Vision & Coaching", 2)
    add_card(
        s2, 0.55, 1.35, 3.85, 5.4, "Injury Risk Under Load",
        [
            "Unsupervised trainees frequently round the lumbar spine during squats/deadlifts, flare elbows during push-ups, or swing the torso on curls.",
            "Errors compound across fatigue sets before the athlete realizes posture has degraded.",
            "Standard webcam tools fail when legs leave frame (chest-up curls/presses) or in side-view occlusion.",
        ],
        RED_ERR,
    )
    add_card(
        s2, 4.74, 1.35, 3.85, 5.4, "Passive Tracker Gap",
        [
            "Existing fitness apps behave as manual notebooks — logging reps only after a set ends without checking technique.",
            "Commercial vision systems struggle in multi-person gyms, attaching stats to passersby or breaking when athletes turn sideways.",
            "Most prototypes evaluate a single exercise via hard-coded heuristics with no cross-subject validation.",
        ],
        AMBER,
    )
    add_card(
        s2, 8.93, 1.35, 3.85, 5.4, "Our Engineering Goal",
        [
            "Zero Hardware Cost: Replace wearable sensors with a single smartphone or laptop RGB camera.",
            "Personalized Biometric Lock: Recognize the enrolled athlete once via SFace embeddings and hold identity across turns.",
            "Actionable Voice Fixes: Convert AI probabilities + biomechanics into concise spoken cues while exercising.",
        ],
        EMERALD,
    )

    # === SLIDE 3: END-TO-END THREE-ENGINE ARCHITECTURE ===
    s3 = prs.slides.add_slide(blank)
    add_header(s3, "02 · System Architecture", "End-to-End Three-Engine Pipeline: See · Know · Coach", 3)
    stages = [
        ("1. INPUT STREAM", ["Smartphone Camera (Flutter)", "Laptop Webcam (Web UI)", "Uploaded Set Video (MP4)"], CYAN),
        ("2. PERCEPTION ENGINE", ["MediaPipe Pose (33x 3D+2D)", "YuNet Face Detect + SFace", "IoU Active-User Lock-On"], EMERALD),
        ("3. AI DECISION ENGINE", ["Stage-1 Auto-Exercise ID", "RF / Ensemble Form Scorer", "SmartFit Fitness Level AI"], AMBER),
        ("4. COACHING & STORAGE", ["TTS Voice Cues + HUD Box", "H.264 Annotated MP4 Replay", "SQLite Dynamic Level Update"], CYAN),
    ]
    x0 = 0.55
    for idx, (st_title, st_items, st_col) in enumerate(stages):
        add_card(s3, x0, 1.45, 2.75, 2.75, st_title, st_items, st_col, body_size=11.5)
        if idx < 3:
            arr = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x0 + 2.83), Inches(2.65), Inches(0.32), Inches(0.36))
            arr.fill.solid()
            arr.fill.fore_color.rgb = EMERALD
            arr.line.fill.background()
        x0 += 3.16

    add_card(
        s3, 0.55, 4.50, 6.0, 2.25, "Real-Time WebSocket Mode (/ws/live/{exercise})",
        [
            "Streams JPEG frames at ~6–10 FPS to FastAPI backend.",
            "Extracts 12-joint normalized skeleton + SFace HUD lock-on + rolling rep phase.",
            "Speaks immediate throttled safety cues (>=3.5s gap) & rep-end feedback via TTS.",
        ],
        EMERALD, body_size=12.0,
    )
    add_card(
        s3, 6.78, 4.50, 6.0, 2.25, "Post-Set Video Audit Mode (/api/analyze/video)",
        [
            "Processes full set MP4 with Stage-1 auto-detection & per-arm/joint segmentation.",
            "Evaluates every rep (Random Forest / Ensemble + biomechanical diagnostics).",
            "Encodes browser/mobile-playable H.264 (libx264, yuv420p, +faststart) MP4.",
        ],
        CYAN, body_size=12.0,
    )

    # === SLIDE 4: COMPUTER VISION & FEATURE ENGINEERING ===
    s4 = prs.slides.add_slide(blank)
    add_header(s4, "03 · Computer Vision & Features", "Scale-Invariant 3D + 2D Representations & Occlusion Resilience", 4)
    add_card(
        s4, 0.55, 1.35, 4.0, 3.15, "Tensor A: Skeleton Grid (32, 12, 6)",
        [
            "12 core upper/lower joints (11–16, 23–28) extracted in metric 3D world space.",
            "Centered at mid-hip & scaled by torso length (or shoulder span x 1.35 when hips off-screen).",
            "Channels: (x, y, z) position + (dx, dy, dz) temporal velocity across T=32 frames.",
        ],
        CYAN, body_size=11.8,
    )
    add_card(
        s4, 4.66, 1.35, 4.0, 3.15, "Tensor B: 14 Biomechanical Angles",
        [
            "Left/Right Elbow (fused 3D+2D for curls!), Shoulder & Hip flexion angles.",
            "Torso Tilt, Shoulder/Hip Line Tilts, Normalized Wrist Heights, Full Body Line.",
            "Left/Right Knee angles -> immune to camera distance and athlete height.",
        ],
        EMERALD, body_size=11.8,
    )
    add_card(
        s4, 8.77, 1.35, 4.0, 3.15, "Phase-Aware Tabular Vector (178-D)",
        [
            "70 Global stats (mean, std, min, max, ROM across 14 channels).",
            "70 Three-Phase stats: Descent [0:11], Bottom/Peak [11:22] mean/min/max, Ascent [22:32].",
            "28 Angular velocity stats (|d/dt| mean & max) + 10 Left-Right asymmetry pairs.",
        ],
        AMBER, body_size=11.8,
    )
    add_card(
        s4, 0.55, 4.72, 12.23, 2.05, "Field-Calibrated Framing Fixes (Solved Real Camera Failure Modes)",
        [
            "Chest-Up Webcam Support: Accepts upper-arm chains (11-13-15 or 12-14-16) even when hips are off-screen; disables invalid lower-back tilt warnings automatically when hips are cropped.",
            "Side-View Occlusion Handling: Tracks whichever camera-facing arm/leg chain passes visibility >=0.28, preventing 0-frame drops.",
            "3D Foreshortening Correction: Bicep curl tracker blends 3D world + 2D image elbow angles per arm to catch full contractions cleanly.",
        ],
        EMERALD, body_size=12.0,
    )

    # === SLIDE 5: ALL 7 EXERCISES + STAGE-1 AUTO-DETECTION ===
    s5 = prs.slides.add_slide(blank)
    add_header(s5, "04 · Multi-Exercise Intelligence", "Stage-1 Auto-Detection & Biomechanics Across All 7 Exercises", 5)
    tbl5 = s5.shapes.add_table(8, 4, Inches(0.55), Inches(1.30), Inches(12.23), Inches(5.45)).table
    style_table(
        tbl5,
        [2.35, 2.55, 3.65, 3.68],
        ["Exercise", "Primary Signal & Framing", "Auto-Detection Signature", "Specific Corrective Voice Cues"],
        [
            ["Squat", "Bilateral Mean Knee (full body)", "Upright torso <48° + symmetric knee & hip flexion ROM >=26°", "Go deeper · Keep chest up · Keep hips level · Align knees"],
            ["Push-Up", "Dominant Elbow (full plank)", "Horizontal plank: torso >58° & hips >128° + elbow ROM >=16°", "Keep back straight & core tight · Lower chest · Tuck elbows"],
            ["Bicep Curl", "Independent L/R Elbow (3D+2D)", "Low shoulders (<58°) + elbow ROM >=24° (barbell & alternating)", "Pin elbows to sides · Curl higher (full ROM) · Avoid swinging"],
            ["Shoulder Press / OHP", "Max Visible Shoulder + Wrist H.", "Wrists above shoulders (>0.22) + shoulder elevation >96°", "Press all the way overhead · Brace core · Press evenly"],
            ["Bent-Over Row / Deadlift", "Dual Sub-Tracker (Elbow & Hip)", "Standing forward hinge (torso 22–76°) + knees >112°", "Pull elbows higher · Hinge forward · Lock out hips at top"],
            ["Lunge", "Min L/R Knee (split stance)", "Knee bend >=26° with hip split asymmetry >=32°", "Drop back knee lower · Keep torso upright"],
            ["Arm Abduction", "Max Shoulder Angle (standing)", "Wide shoulder raise >=62° with straight elbows & wrists <=0.20", "Raise arms to shoulder height · Avoid leaning · Soft elbow"],
        ],
        font_size=11.0,
    )

    # === SLIDE 6: DATASETS & VALIDATION PROTOCOL ===
    s6 = prs.slides.add_slide(blank)
    add_header(s6, "05 · Datasets & Zero-Leakage Validation", "Five Benchmarks Unified Under Strict Subject-Grouped CV", 6)
    add_card(
        s6, 0.55, 1.35, 6.0, 2.65, "Mendeley + REHAB24-6 + UI-PRMD (Form Quality)",
        [
            "Mendeley kgbb3yn47p (v3): 25–26 subjects x 3 camera views (front, side, diagonal) -> Push-Up, Curl (2,175 reps), Shoulder Press (1,961 reps).",
            "REHAB24-6 (Zenodo 13305826) + UI-PRMD (Zenodo 23104965): merged 3D Kinect/RGB rehab & gym skeletons -> Squat (232 reps), Lunge (18 subj, 375 reps), Arm Abduction (19 subj, 378 reps).",
            "Unified 14-angle + 178-D phase feature pipeline aligns RGB & 3D Kinect skeletons seamlessly.",
        ],
        EMERALD, body_size=11.4,
    )
    add_card(
        s6, 6.78, 1.35, 6.0, 2.65, "SmartFit Profiles & LFW Biometrics",
        [
            "SmartFit Dataset (Tomertg/SmartFit_Dataset): 1,000 athlete profiles (Age, Height, Weight, BMI, Gender -> Beginner/Intermediate/Advanced).",
            "LFW (Labeled Faces in the Wild): 1,000 unseen verification pairs (500 genuine same-person, 500 impostor) for biometric threshold tuning.",
            "Kept strictly decoupled from pose tensors to prevent synthetic noise leakage.",
        ],
        CYAN, body_size=11.8,
    )
    add_card(
        s6, 0.55, 4.25, 12.23, 2.55, "Why StratifiedGroupKFold (5-Fold by subject_id) Matters",
        [
            "Zero Subject Leakage: Every repetition from the same participant stays strictly in train OR test within each fold — evaluating true generalization to unseen new users.",
            "Training-Only Augmentation: Physically valid horizontal mirror flips (swapping L/R joints + negating X/dx) and mild Gaussian landmark jitter are applied ONLY inside training folds.",
            "Comprehensive Metrics: Primary = Out-of-Fold (OOF) Macro F1; Supporting = Balanced Acc, ROC-AUC, PR-AUC, Model Size (KB), and Single-Rep CPU Latency (ms).",
        ],
        AMBER, body_size=12.0,
    )

    # === SLIDE 7: 10-MODEL BENCHMARK ARCHITECTURES ===
    s7 = prs.slides.add_slide(blank)
    add_header(s7, "06 · Model Architectures", "Four Architectural Families Benchmarked Side-by-Side (10 Models)", 7)
    add_card(
        s7, 0.55, 1.35, 6.0, 2.60, "Family 1: Classical / Tabular ML (4 Models)",
        [
            "Input: 178-D Phase-Aware + Velocity + L/R Asymmetry Feature Vector.",
            "Random Forest (250 trees, max_depth=12, balanced class weights).",
            "LightGBM (200 trees) · XGBoost (scale_pos_weight) · SVM (RBF, C=2.0, Platt scaling).",
        ],
        EMERALD, body_size=12.0,
    )
    add_card(
        s7, 6.78, 1.35, 6.0, 2.60, "Family 2: Deep Learning Sequence & Grid (4 Models)",
        [
            "1D CNN & Dilated TCN (dilation 1, 2, 4 + residual blocks) on (32, 14) angle curves.",
            "BiLSTM + Temporal Attention Pooling: learns which rep frames carry form errors.",
            "2D Skeleton CNN: Conv2D + SpatialDropout2D over (32, 12, 6) body+velocity grid.",
        ],
        CYAN, body_size=12.0,
    )
    add_card(
        s7, 0.55, 4.20, 6.0, 2.55, "Family 3: Merged Two-Stream Hybrid (#9)",
        [
            "Merged 2D-CNN + BiGRU fuses BOTH representations end-to-end:",
            "Stream A (Spatial-Temporal): Conv2D layers over (32, 12, 6) skeleton grid.",
            "Stream B (Kinematic): Conv1D + BiGRU + TemporalAttentionPool over (32, 14) angles.",
        ],
        AMBER, body_size=12.0,
    )
    add_card(
        s7, 6.78, 4.20, 6.0, 2.55, "Family 4: Cross-Family Soft-Voting Ensemble (#10)",
        [
            "Automatically blends Best Deep Model (weight 0.55) + Best Tabular Model (weight 0.45).",
            "Out-of-fold probability fusion reduces variance on unseen subjects and camera angles.",
            "Stability hardening: Adam(clipnorm=1.0) + NaN-safe inference sanitization.",
        ],
        AMBER, body_size=12.0,
    )

    # === SLIDE 8: SQUAT RESULTS ===
    s8 = prs.slides.add_slide(blank)
    add_header(s8, "07 · Experimental Results — Squat", "Upgraded Squat Benchmark (UI-PRMD · Unified Ensemble #1)", 8)
    s8.shapes.add_picture(str(squat_png), Inches(0.55), Inches(1.28), width=Inches(8.35))
    add_kpi_tile(s8, 9.15, 1.30, 3.63, 1.05, "0.9267 F1", "#1 Winner: RF + HistGB + SVM", EMERALD)
    add_kpi_tile(s8, 9.15, 2.50, 3.63, 1.05, "92.7% Acc", "0.9204 ROC-AUC (Unseen Subjects)", CYAN)
    add_card(
        s8, 9.15, 3.75, 3.63, 3.00, "Key Takeaways (Squat)",
        [
            "Soft-Voting Ensemble (RF+HistGB+SVM) ranked #1 at 0.9267 F1 (+1.13% over single RF 0.9154).",
            "Upgraded from 0.6656 F1 (earlier 35-subj Mendeley+REHAB baseline) via UI-PRMD 3D kinematics.",
            "Pure scikit-learn CPU ensemble runs live with zero TensorFlow overhead.",
        ],
        AMBER, body_size=11.2,
    )

    # === SLIDE 9: PUSH-UP RESULTS ===
    s9 = prs.slides.add_slide(blank)
    add_header(s9, "08 · Experimental Results — Push-Up", "Push-Up 10-Model Leaderboard: Cross-Family Ensemble Wins", 9)
    s9.shapes.add_picture(str(push_png), Inches(0.55), Inches(1.28), width=Inches(8.35))
    add_kpi_tile(s9, 9.15, 1.30, 3.63, 1.05, "0.6606", "#1 Winner: 2D-CNN + BiGRU + RF", AMBER)
    add_kpi_tile(s9, 9.15, 2.50, 3.63, 1.05, "0.7061 ROC-AUC", "66.7% Acc · Outperforms Single Models", CYAN)
    add_card(
        s9, 9.15, 3.75, 3.63, 3.00, "Key Takeaways (Push-Up)",
        [
            "Fusing Two-Stream 2D-CNN + BiGRU (0.55) with Random Forest (0.45) lifted Macro F1 by +0.0162 over the top single model.",
            "Random Forest (#2, 0.6444 F1, 39.5 ms) stays under the 50 ms real-time budget.",
            "LightGBM (1.78 ms) & XGBoost (12.05 ms) confirm ultra-fast tree inference.",
        ],
        EMERALD, body_size=11.5,
    )

    # === SLIDE 10: FINAL MODEL SELECTION & ALL-6 DEPLOYMENT ===
    s10 = prs.slides.add_slide(blank)
    add_header(s10, "09 · Final Model Selection", "Unified Soft-Voting Ensemble (RF + HistGB + SVM) Deployed Live", 10)
    s10.shapes.add_picture(str(all6_rf_png), Inches(0.55), Inches(1.28), width=Inches(7.70))
    add_card(
        s10, 8.42, 1.28, 4.36, 2.75, "Why Unified Soft-Voting Ensemble?",
        [
            "Combines 3 diverse families (Random Forest + HistGradientBoosting + SVM-RBF) in pure scikit-learn.",
            "Beat single RF on Curl (0.9771), Squat (0.9267), and Lunge (0.8283, +11.6% F1 jump!) + tied Abduction (0.8894).",
            "Push-Up (0.6444 RF / 0.6606 Deep+RF) & Shoulder (0.5873 RF) run live on CPU.",
        ],
        EMERALD, body_size=11.2,
    )
    add_card(
        s10, 8.42, 4.20, 4.36, 2.56, "Per-Exercise vs. Single Universal Model",
        [
            "We also tested 1 Universal Model across all exercises (0.9055 overall F1).",
            "Per-exercise Unified Ensemble beat it by +9.25% on Lunge (0.8283 vs 0.7358) & +3.45% on Curl (0.9771 vs 0.9426).",
            "Proves 1 unified architecture with per-exercise weights maximizes accuracy.",
        ],
        AMBER, body_size=11.2,
    )

    # === SLIDE 11: BIOMETRIC ATHLETE IDENTITY ===
    s11 = prs.slides.add_slide(blank)
    add_header(s11, "10 · Biometric Face ID & HUD", "3-Angle Face ID Enrollment & Sideways-Resilient Body Lock-On", 11)
    add_card(
        s11, 0.55, 1.35, 6.0, 2.75, "3-Angle Live Enrollment: OpenCV YuNet + SFace",
        [
            "Guided Face ID scanner captures 3 poses (Forward · Left · Right) with voice prompts & auto-countdown right on localhost.",
            "SFace (face_recognition_sface_2021dec.onnx) extracts a multi-angle (K x 128) L2-normalized template stored in SQLite.",
            "Matches via max cosine similarity (>= 0.363) across all stored angles for reliable side-turn recognition.",
        ],
        CYAN, body_size=11.8,
    )
    add_card(
        s11, 6.78, 1.35, 6.0, 2.75, "ActiveUserTracker: Surviving Turns & Gym Crowds",
        [
            "Locks recognized athlete's Name, Age, Height & Fitness Level onto their pose bounding box.",
            "Crops upper-body ROI every 6 frames with IoU (>0.15) continuity & 25s TTL.",
            "HUD stays locked even when the athlete turns sideways for deadlifts, rows, or push-ups.",
        ],
        EMERALD, body_size=12.0,
    )
    add_card(
        s11, 0.55, 4.35, 12.23, 2.40, "Multi-Person Privacy & Edge Architecture Highlights",
        [
            "Privacy-First Local Storage: Only 128-float embeddings + cropped registration thumbnails live in local SQLite (backend/data/gym_trainer.db); raw camera frames are never uploaded to third-party servers.",
            "Largest-Face Priority + Upper-Box Cropping: Prevents background gym passersby from hijacking the active athlete's rep counter or HUD card.",
        ],
        AMBER, body_size=12.0,
    )

    # === SLIDE 12: SMARTFIT PROFILE AI ===
    s12 = prs.slides.add_slide(blank)
    add_header(s12, "11 · Athlete Profile AI", "SmartFit Fitness-Level Predictor & Dynamic Longitudinal Progression", 12)
    s12.shapes.add_picture(str(smartfit_png), Inches(0.55), Inches(1.32), width=Inches(7.35))
    add_card(
        s12, 8.12, 1.32, 4.66, 2.55, "Signup Personalization (SmartFit AI)",
        [
            "Winner: Random Forest (0.4589 Macro F1 across 5-fold CV over SVM-RBF 0.4555 & MLP 0.4521).",
            "Inputs: Age, Height (cm), Weight (kg), derived BMI, Gender.",
            "Outputs Level + adaptive prescription (sets, target reps, rest interval).",
        ],
        EMERALD, body_size=11.8,
    )
    add_card(
        s12, 8.12, 4.05, 4.66, 2.70, "Dynamic Level Adaptation (SQLite)",
        [
            "Every live or post-set workout logs reps, form_score, cues & video_url.",
            "After >=3 sessions, rolling average form score dynamically promotes/adapts the user:",
            "Avg >=85% -> Advanced | >=65% -> Intermediate | <65% -> Beginner.",
        ],
        CYAN, body_size=11.8,
    )

    # === SLIDE 13: EXPLAINABLE VOICE COACHING & STACK ===
    s13 = prs.slides.add_slide(blank)
    add_header(s13, "12 · XAI Coaching & Software Stack", "Explainable Biomechanical Cues, H.264 Playback & Full-Stack App", 13)
    add_card(
        s13, 0.55, 1.35, 3.85, 5.4, "Explainable Voice Coach (TTS)",
        [
            "Instead of opaque probabilities alone, rule-based kinematics pinpoint WHICH joint fault occurred.",
            "Mid-rep safety alerts throttled every >=3.5s so coaching never overwhelms.",
            "Delivered via Web Speech API (browser) and flutter_tts (mobile).",
        ],
        EMERALD, body_size=12.0,
    )
    add_card(
        s13, 4.74, 1.35, 3.85, 5.4, "FastAPI Backend & H.264 Fix",
        [
            "REST API: /api/health, /api/users/register, /api/identify, /api/analyze/video.",
            "WebSocket: /ws/live/{exercise} streams JSON landmarks + rep counts.",
            "Browser Replay Fix: Re-encodes OpenCV output via bundled ffmpeg (libx264, yuv420p, +faststart -> avc1 atom).",
        ],
        CYAN, body_size=12.0,
    )
    add_card(
        s13, 8.93, 1.35, 3.85, 5.4, "Web Demo + FormCoach Flutter",
        [
            "Built-in 3-Tab Web Demo UI for zero-install defense presentation.",
            "FormCoach Flutter Mobile App fixed & wired via BackendPoseAnalyzer & BackendConfig.",
            "Supports all 7 exercises + Auto-Detect + Android HTTP/WS permissions.",
        ],
        AMBER, body_size=12.0,
    )

    # === SLIDE 14: VERIFICATION & VISUAL SHOWCASE ===
    s14 = prs.slides.add_slide(blank)
    add_header(s14, "13 · End-to-End Verification", "Verified Across All 7 Exercises, Chest-Up Webcam Crops & H.264 Replay", 14)
    s14.shapes.add_picture(str(collage_png), Inches(0.55), Inches(1.22), width=Inches(12.23))
    add_kpi_tile(s14, 0.55, 5.15, 2.85, 1.55, "12 / 12", "Real Exercise Clips Passed\n(Full + Chest-Up Webcam)", EMERALD)
    add_kpi_tile(s14, 3.68, 5.15, 2.85, 1.55, "7 / 7", "Stage-1 Auto-Detection\n(Idle-Protected Rolling Window)", CYAN)
    add_kpi_tile(s14, 6.81, 5.15, 2.85, 1.55, "4 / 4 Reps", "Alternating & Bilateral\nBicep Curl L/R Segmentation", AMBER)
    add_kpi_tile(s14, 9.94, 5.15, 2.90, 1.55, "avc1 H.264", "Post-Set Video Verified\nPlayable in Chrome/Flutter", EMERALD)

    # === SLIDE 15: RESEARCH QUESTIONS ANSWERED ===
    s15 = prs.slides.add_slide(blank)
    add_header(s15, "14 · Academic Evaluation", "Proposal Research Questions Mapped to Implemented Evidence", 15)
    tbl15 = s15.shapes.add_table(6, 3, Inches(0.55), Inches(1.30), Inches(12.23), Inches(5.45)).table
    style_table(
        tbl15,
        [1.15, 4.75, 6.33],
        ["RQ", "Original Proposal Question", "Implemented Solution & Verified Finding"],
        [
            ["RQ1", "Can a single smartphone/webcam replace lab rigs for 3D pose kinematics?", "YES — MediaPipe Pose 3D+2D + torso scaling tracks 7 exercises cleanly; 3D+2D elbow fusion solves front-view curl foreshortening."],
            ["RQ2", "Which model family maximizes cross-subject form accuracy under real-time limits?", "Unified Random Forest wins Lunge (0.7774) & Curl (0.6258), hits 0.9056 F1 on UI-PRMD Squat & 0.7495 on Abduction; 2D-CNN+BiGRU+RF Ensemble wins Push-Up (0.6606)."],
            ["RQ3", "How can vision connect with athlete history & adaptive prescriptions?", "SmartFit Random Forest predicts baseline fitness tier at signup; SQLite logs every session and dynamically adapts user level after >=3 sets."],
            ["RQ4", "Why edge/local vision processing over cloud-only video uploads?", "Local FastAPI WebSocket keeps latency responsive without sending raw biometric video externally; post-set audit runs locally too."],
            ["RQ5", "How do we hold identity lock across multi-person gyms & sideways turns?", "3-Angle YuNet + SFace enrollment (Forward/Left/Right) + max-cosine matching & ActiveUserTracker IoU lock hold identity through turns."],
        ],
        font_size=11.0,
    )

    # === SLIDE 16: CONTRIBUTIONS, LIMITATIONS & Q&A ===
    s16 = prs.slides.add_slide(blank)
    add_header(s16, "15 · Conclusion & Future Work", "Key Contributions, Honest Scope Limits & Live Demo Ready", 16)
    add_card(
        s16, 0.55, 1.35, 6.0, 3.65, "Core Project Contributions",
        [
            "Unified Multi-Dataset Benchmark: Leakage-free CV across Mendeley, REHAB24-6 & UI-PRMD with 6/7 exercises deployed on trained Random Forest models.",
            "7-Exercise Live + Auto-Detect Engine: Robust to chest-up laptop webcams, alternating curls, and side-view occlusions.",
            "Complete Ecosystem: Working FastAPI backend, 3-Angle YuNet+SFace HUD, SmartFit AI, H.264 video reporter, Web Demo, and connected Flutter app.",
        ],
        EMERALD, body_size=12.0,
    )
    add_card(
        s16, 6.78, 1.35, 6.0, 3.65, "Honest Limitations & Next Phase",
        [
            "6 of 7 exercises + SmartFit run trained Random Forest models; Bent-Over Row currently uses calibrated biomechanical rules pending labeled row video collection.",
            "Monocular depth ambiguity under extreme occlusion (barbell bench racks) remains out-of-scope.",
            "Next step: TFLite / ONNX Mobile export inside Flutter for 100% offline phone inference.",
        ],
        AMBER, body_size=12.0,
    )
    banner = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(5.25), Inches(12.23), Inches(1.50))
    banner.adjustments[0] = 0.08
    banner.fill.solid()
    banner.fill.fore_color.rgb = CARD_ALT
    banner.line.color.rgb = EMERALD
    banner.line.width = Pt(1.5)
    btf = banner.text_frame
    btf.word_wrap = True
    btf.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp1 = btf.paragraphs[0]
    bp1.alignment = PP_ALIGN.CENTER
    bp1.text = "Thank You — Questions & Live System Demonstration"
    bp1.font.name = FONT_FAMILY
    bp1.font.size = Pt(24)
    bp1.font.bold = True
    bp1.font.color.rgb = TEXT_MAIN
    bp2 = btf.add_paragraph()
    bp2.alignment = PP_ALIGN.CENTER
    bp2.text = "Live Web Demo: http://localhost:8000   |   Swagger API: /docs   |   Mobile Client: FormCoach Flutter"
    bp2.font.name = FONT_FAMILY
    bp2.font.size = Pt(13)
    bp2.font.color.rgb = CYAN

    prs.save(str(OUT_PPTX))
    check_prs = Presentation(str(OUT_PPTX))
    print(f"Saved {OUT_PPTX} | Slides: {len(check_prs.slides)} | Size: {OUT_PPTX.stat().st_size / 1024:.1f} KB")
    assert len(check_prs.slides) == TOTAL_SLIDES


if __name__ == "__main__":
    build_deck()
