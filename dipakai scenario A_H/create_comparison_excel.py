"""
Membuat Excel perbandingan hasil fitness dari berbagai skenario A-H
CL-GWO dengan CNN (ConvNeXt-Tiny), dataset vegetation
"""

import json
import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.utils.dataframe import dataframe_to_rows

BASE_DIR = "/mnt/d/Hibah BIMA/2026/BIMA_RAP/full_code_RAP/dipakai scenario A_H"

# Best params per scenario
BEST = {
    "A": {"pop": 90, "iter": 60, "folder": "cnn_CL_GWO_scenA_hybrid_pop90_iter60_pop90_iter60_20260602_084635"},
    "B": {"pop": 50, "iter": 60, "folder": "cnn_CL_GWO_scenB_hybrid_pop50_iter60_pop50_iter60_20260602_143254"},
    "C": {"pop": 80, "iter": 80, "folder": "cnn_CL_GWO_scenC_hybrid_pop80_iter80_pop80_iter80_20260602_204336"},
    "D": {"pop": 90, "iter": 30, "folder": "cnn_CL_GWO_scenD_hybrid_pop90_iter30_pop90_iter30_20260603_003507"},
    "E": {"pop":100, "iter": 10, "folder": "cnn_CL_GWO_scenE_hybrid_pop100_iter10_pop100_iter10_20260603_071956"},
    "F": {"pop": 10, "iter": 10, "folder": "cnn_CL_GWO_scenF_hybrid_pop10_iter10_pop10_iter10_20260603_175702"},
    "G": {"pop": 70, "iter":100, "folder": "cnn_CL_GWO_scenG_hybrid_pop70_iter100_pop70_iter100_20260604_024655"},
    "H": {"pop": 70, "iter": 10, "folder": "cnn_CL_GWO_scenH_hybrid_pop70_iter10_pop70_iter10_20260604_042438"},
}

SCENARIO_LABELS = {
    "A": "Fitness = Akurasi(selected+f_selected)",
    "B": "Fitness = Akurasi(discarded+f_discarded)",
    "C": "Fitness = Akurasi(all+f_selected)",
    "D": "Fitness = Akurasi(all+f_discarded)",
    "E": "Fitness = Akurasi(all+f_sel+f_disc)",
    "F": "Fitness = Akurasi(all_features_only)",
    "G": "Fitness = Akurasi(selected_only)",
    "H": "Fitness = Akurasi(discarded_only)",
}

EVAL_KEYS_ORDER = [
    "A_selected_plus_f_selected",
    "B_discarded_plus_f_discarded",
    "C_all_plus_f_selected",
    "D_all_plus_f_discarded",
    "E_all_plus_f_selected_plus_f_discarded",
    "F_all_features_only",
    "G_selected_only",
    "H_discarded_only",
]

EVAL_LABELS = {
    "A_selected_plus_f_selected":           "A: Selected + f_selected",
    "B_discarded_plus_f_discarded":          "B: Discarded + f_discarded",
    "C_all_plus_f_selected":                 "C: All + f_selected",
    "D_all_plus_f_discarded":                "D: All + f_discarded",
    "E_all_plus_f_selected_plus_f_discarded":"E: All + f_sel + f_disc",
    "F_all_features_only":                   "F: All features only",
    "G_selected_only":                       "G: Selected only",
    "H_discarded_only":                      "H: Discarded only",
}

with open(os.path.join(BASE_DIR, "best_params_CLGWO_all_scenarios.json")) as f:
    best_params = json.load(f)

# ── Load all data ──────────────────────────────────────────────────────────────
scenario_data = {}   # sn -> {"final": df, "fold": df, "opt_best": series, "opt_hist": df}

for sn, info in BEST.items():
    folder_path = os.path.join(BASE_DIR, f"results_CLGWO_scen{sn}", info["folder"])
    final = pd.read_csv(os.path.join(folder_path, "final_classification_results.csv"))
    fold  = pd.read_csv(os.path.join(folder_path, "fold_classification_results.csv"))
    opt_best = pd.read_csv(os.path.join(folder_path, "optimization_history_best.csv"))
    opt_hist = pd.read_csv(os.path.join(folder_path, "optimization_history.csv"))
    scenario_data[sn] = {
        "final": final,
        "fold":  fold,
        "opt_best": opt_best,
        "opt_hist": opt_hist,
    }

# ── Styles ─────────────────────────────────────────────────────────────────────
HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")
SUBHDR_FILL   = PatternFill("solid", fgColor="2E75B6")
SCEN_FILLS = {
    "A": PatternFill("solid", fgColor="D6E4F0"),
    "B": PatternFill("solid", fgColor="D5E8D4"),
    "C": PatternFill("solid", fgColor="FFF2CC"),
    "D": PatternFill("solid", fgColor="FCE4D6"),
    "E": PatternFill("solid", fgColor="EAD1DC"),
    "F": PatternFill("solid", fgColor="E2EFDA"),
    "G": PatternFill("solid", fgColor="DDEBF7"),
    "H": PatternFill("solid", fgColor="F4CCCC"),
}
HEADER_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
SUBHDR_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
BOLD_FONT     = Font(name="Calibri", bold=True, size=10)
NORMAL_FONT   = Font(name="Calibri", size=10)
CENTER        = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT          = Alignment(horizontal="left",   vertical="center", wrap_text=True)

thin = Side(style="thin", color="BFBFBF")
THIN_BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

thick_side = Side(style="medium", color="1F4E79")
THICK_BORDER = Border(left=thick_side, right=thick_side, top=thick_side, bottom=thick_side)

PCT_FMT = "0.00%"
NUM_FMT = "0.0000"

def style_cell(cell, fill=None, font=None, align=CENTER, border=THIN_BORDER, num_format=None):
    if fill:   cell.fill   = fill
    if font:   cell.font   = font or NORMAL_FONT
    if align:  cell.alignment = align
    if border: cell.border = border
    if num_format: cell.number_format = num_format

def write_header(ws, row, col, text, fill=HEADER_FILL, font=HEADER_FONT,
                 merge_cols=1, merge_rows=1):
    cell = ws.cell(row=row, col=col) if False else ws.cell(row=row, column=col)
    cell.value = text
    style_cell(cell, fill=fill, font=font)
    if merge_cols > 1 or merge_rows > 1:
        ws.merge_cells(start_row=row, start_column=col,
                       end_row=row+merge_rows-1, end_column=col+merge_cols-1)
    return cell


# ══════════════════════════════════════════════════════════════════════════════
# WORKBOOK
# ══════════════════════════════════════════════════════════════════════════════
wb = Workbook()

# ─────────────────────────────────────────────────────────────────────────────
# SHEET 1 : Ringkasan – Akurasi terbaik vs eval scenario
# ─────────────────────────────────────────────────────────────────────────────
ws1 = wb.active
ws1.title = "Ringkasan Perbandingan"

ws1.row_dimensions[1].height = 40
ws1.row_dimensions[2].height = 30
ws1.row_dimensions[3].height = 20

# Title
ws1.merge_cells("A1:M1")
title_cell = ws1["A1"]
title_cell.value = "PERBANDINGAN HASIL FITNESS BERBAGAI SKENARIO — CL-GWO (ConvNeXt-Tiny)"
style_cell(title_cell, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=14))

ws1.merge_cells("A2:M2")
sub_cell = ws1["A2"]
sub_cell.value = (
    "Dataset: Type of Vegetation  |  Backbone: ConvNeXt-Tiny  |  "
    "Meta-Algo: CL-GWO  |  CV: 5-Fold  |  CNN Epochs: 30"
)
style_cell(sub_cell, fill=SUBHDR_FILL, font=Font(name="Calibri", bold=False, color="FFFFFF", size=10))

# Header row 3
headers_r3 = [
    "Fitness\nSkenario",
    "Deskripsi Fitness",
    "Best\nPop",
    "Best\nIter",
    "Best Fitness\n(val akurasi optim.)",
    "Waktu\n(menit)",
]
# Then 8 eval columns
for ek in EVAL_KEYS_ORDER:
    headers_r3.append(f"Eval\n{EVAL_LABELS[ek]}")
headers_r3.append("Jumlah\nFitur Terpilih")

for c, h in enumerate(headers_r3, 1):
    cell = ws1.cell(row=3, column=c, value=h)
    style_cell(cell, fill=SUBHDR_FILL, font=SUBHDR_FONT)

# set column widths
col_widths = [8, 40, 8, 8, 20, 12] + [18]*8 + [14]
for c, w in enumerate(col_widths, 1):
    ws1.column_dimensions[get_column_letter(c)].width = w

# Data rows
for r, sn in enumerate(["A","B","C","D","E","F","G","H"], 4):
    ws1.row_dimensions[r].height = 22
    info = BEST[sn]
    bp = best_params[sn]
    data = scenario_data[sn]
    final_df = data["final"].set_index("scenario")
    opt_best_df = data["opt_best"]
    best_fit_acc = opt_best_df["accuracy"].iloc[-1]  # final best accuracy in optim

    fill = SCEN_FILLS[sn]

    row_vals = [
        f"Skenario {sn}",
        SCENARIO_LABELS[sn],
        bp["best_pop"],
        bp["best_iter"],
        best_fit_acc,
        round(bp["elapsed_mnt"], 1),
    ]
    for ek in EVAL_KEYS_ORDER:
        if ek in final_df.index:
            row_vals.append(final_df.loc[ek, "mean_accuracy"])
        else:
            row_vals.append(None)

    # num_features of the scenario's own eval key
    own_key = bp["eval_key"]
    if own_key in final_df.index:
        row_vals.append(int(final_df.loc[own_key, "num_features"]))
    else:
        row_vals.append(None)

    for c, val in enumerate(row_vals, 1):
        cell = ws1.cell(row=r, column=c, value=val)
        style_cell(cell, fill=fill, font=NORMAL_FONT, border=THIN_BORDER)
        if c in (5,) or c >= 7:
            cell.number_format = "0.0000"
        if c == 1:
            cell.font = Font(name="Calibri", bold=True, size=10)

# Conditional formatting: color scale on accuracy columns (cols 7-14)
for col_idx in range(7, 15):
    col_letter = get_column_letter(col_idx)
    ws1.conditional_formatting.add(
        f"{col_letter}4:{col_letter}11",
        ColorScaleRule(
            start_type="min", start_color="FF6B6B",
            mid_type="percentile", mid_value=50, mid_color="FFD700",
            end_type="max", end_color="57BB8A"
        )
    )

# Bold highlight: each skenario highlights its own fitness evaluation column
OWN_EVAL_COL = {
    "A": 7,   # A_selected_plus_f_selected = col 7
    "B": 8,
    "C": 9,
    "D": 10,
    "E": 11,
    "F": 12,
    "G": 13,
    "H": 14,
}
for r, sn in enumerate(["A","B","C","D","E","F","G","H"], 4):
    own_col = OWN_EVAL_COL[sn]
    cell = ws1.cell(row=r, column=own_col)
    cell.font = Font(name="Calibri", bold=True, size=11, color="1F4E79")
    cell.fill = PatternFill("solid", fgColor="BDD7EE")

# Freeze panes
ws1.freeze_panes = "C4"


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 2 : Detail per Skenario (Akurasi ± StdDev semua eval)
# ─────────────────────────────────────────────────────────────────────────────
ws2 = wb.create_sheet("Detail Akurasi & StdDev")
ws2.row_dimensions[1].height = 36
ws2.row_dimensions[2].height = 28
ws2.row_dimensions[3].height = 22

col_count = 2 + 3 * 8  # Eval Scenario | Eval Label | (mean, std, n_feat) x8
ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=col_count)
tc = ws2.cell(row=1, column=1, value="DETAIL AKURASI DAN STANDAR DEVIASI PER SKENARIO EVALUASI")
style_cell(tc, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=13))

# Row 2: Fitness scenario headers (groups of 3)
ws2.cell(row=2, column=1, value="Evaluasi Scenario")
ws2.cell(row=3, column=1, value="Kode")
ws2.cell(row=2, column=2, value="Deskripsi Evaluasi")
ws2.cell(row=3, column=2, value="Label Evaluasi")
style_cell(ws2.cell(row=2, column=1), fill=SUBHDR_FILL, font=SUBHDR_FONT)
style_cell(ws2.cell(row=2, column=2), fill=SUBHDR_FILL, font=SUBHDR_FONT)
style_cell(ws2.cell(row=3, column=1), fill=SUBHDR_FILL, font=SUBHDR_FONT)
style_cell(ws2.cell(row=3, column=2), fill=SUBHDR_FILL, font=SUBHDR_FONT)
ws2.merge_cells(start_row=2, start_column=1, end_row=3, end_column=1)
ws2.merge_cells(start_row=2, start_column=2, end_row=3, end_column=2)

for i, sn in enumerate(["A","B","C","D","E","F","G","H"]):
    start_col = 3 + i * 3
    # row 2 merged header
    ws2.merge_cells(start_row=2, start_column=start_col, end_row=2, end_column=start_col+2)
    hcell = ws2.cell(row=2, column=start_col, value=f"Fitness Skenario {sn}\n(pop={BEST[sn]['pop']}, iter={BEST[sn]['iter']})")
    style_cell(hcell, fill=SCEN_FILLS[sn], font=Font(name="Calibri", bold=True, size=10))
    # row 3: Mean, Std, N_Feat
    for j, sub in enumerate(["Mean Acc", "Std Acc", "Jml Fitur"]):
        sc = ws2.cell(row=3, column=start_col+j, value=sub)
        style_cell(sc, fill=SCEN_FILLS[sn], font=Font(name="Calibri", bold=True, size=9))

# column widths
ws2.column_dimensions["A"].width = 8
ws2.column_dimensions["B"].width = 36
for i in range(8):
    for j in range(3):
        cl = get_column_letter(3 + i*3 + j)
        ws2.column_dimensions[cl].width = 11

# Data rows
for r, ek in enumerate(EVAL_KEYS_ORDER, 4):
    ws2.row_dimensions[r].height = 20
    ws2.cell(row=r, column=1, value=ek.split("_")[0])
    style_cell(ws2.cell(row=r, column=1), font=BOLD_FONT, border=THIN_BORDER)
    ws2.cell(row=r, column=2, value=EVAL_LABELS[ek])
    style_cell(ws2.cell(row=r, column=2), align=LEFT, font=NORMAL_FONT, border=THIN_BORDER)

    for i, sn in enumerate(["A","B","C","D","E","F","G","H"]):
        final_df = scenario_data[sn]["final"].set_index("scenario")
        fill = SCEN_FILLS[sn]
        start_col = 3 + i * 3
        if ek in final_df.index:
            mean_acc = final_df.loc[ek, "mean_accuracy"]
            std_acc  = final_df.loc[ek, "std_accuracy"]
            n_feat   = int(final_df.loc[ek, "num_features"])
        else:
            mean_acc = std_acc = n_feat = None

        c_mean = ws2.cell(row=r, column=start_col,   value=mean_acc)
        c_std  = ws2.cell(row=r, column=start_col+1, value=std_acc)
        c_nf   = ws2.cell(row=r, column=start_col+2, value=n_feat)
        style_cell(c_mean, fill=fill, font=NORMAL_FONT, border=THIN_BORDER, num_format="0.0000")
        style_cell(c_std,  fill=fill, font=NORMAL_FONT, border=THIN_BORDER, num_format="0.0000")
        style_cell(c_nf,   fill=fill, font=NORMAL_FONT, border=THIN_BORDER)

        # Highlight if own scenario
        own_key = best_params[sn]["eval_key"]
        if ek == own_key:
            c_mean.font = Font(name="Calibri", bold=True, size=11, color="1F4E79")
            c_mean.fill = PatternFill("solid", fgColor="BDD7EE")

ws2.freeze_panes = "C4"


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 3 : Per-fold akurasi (best eval) per skenario
# ─────────────────────────────────────────────────────────────────────────────
ws3 = wb.create_sheet("Per-Fold Akurasi")
ws3.row_dimensions[1].height = 36
ws3.row_dimensions[2].height = 22

ws3.merge_cells("A1:K1")
tc3 = ws3.cell(row=1, column=1, value="AKURASI PER FOLD — EVALUASI SESUAI FITNESS SKENARIO (BEST EVAL KEY)")
style_cell(tc3, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=13))

headers = ["Fitness\nSkenario", "Eval Key (Best)", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5",
           "Mean Acc", "Std Acc", "Min Fold", "Max Fold"]
for c, h in enumerate(headers, 1):
    cell = ws3.cell(row=2, column=c, value=h)
    style_cell(cell, fill=SUBHDR_FILL, font=SUBHDR_FONT)

col_widths3 = [12, 38, 10, 10, 10, 10, 10, 12, 12, 12, 12]
for c, w in enumerate(col_widths3, 1):
    ws3.column_dimensions[get_column_letter(c)].width = w

for r, sn in enumerate(["A","B","C","D","E","F","G","H"], 3):
    ws3.row_dimensions[r].height = 20
    own_key = best_params[sn]["eval_key"]
    fold_df = scenario_data[sn]["fold"]
    fold_rows = fold_df[fold_df["scenario"] == own_key].sort_values("fold")
    accs = list(fold_rows["accuracy"])
    while len(accs) < 5:
        accs.append(None)

    final_df = scenario_data[sn]["final"].set_index("scenario")
    mean_acc = final_df.loc[own_key, "mean_accuracy"] if own_key in final_df.index else None
    std_acc  = final_df.loc[own_key, "std_accuracy"]  if own_key in final_df.index else None

    fill = SCEN_FILLS[sn]
    row_data = [f"Skenario {sn}", own_key] + accs[:5] + [
        mean_acc, std_acc,
        min(a for a in accs if a) if any(accs) else None,
        max(a for a in accs if a) if any(accs) else None,
    ]
    for c, val in enumerate(row_data, 1):
        cell = ws3.cell(row=r, column=c, value=val)
        style_cell(cell, fill=fill, font=NORMAL_FONT, border=THIN_BORDER)
        if c >= 3:
            cell.number_format = "0.0000"
        if c == 1:
            cell.font = Font(name="Calibri", bold=True, size=10)

ws3.freeze_panes = "C3"


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 4 : Optimasi konvergensi ringkasan (iteration, best fitness, elapsed)
# ─────────────────────────────────────────────────────────────────────────────
ws4 = wb.create_sheet("Konvergensi Optimasi")
ws4.row_dimensions[1].height = 36
ws4.row_dimensions[2].height = 22

ws4.merge_cells("A1:I1")
tc4 = ws4.cell(row=1, column=1,
               value="RINGKASAN KONVERGENSI OPTIMASI CL-GWO — BEST PARAMETER PER SKENARIO")
style_cell(tc4, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=13))

headers4 = [
    "Fitness\nSkenario", "Best Pop", "Best Iter",
    "Iter Ditemukan\nBest Fitness", "Best Fitness\n(akurasi optim.)",
    "Waktu Total\n(menit)", "Waktu/Iterasi\n(detik)",
    "Jumlah Fitur\nTerpilih", "Jumlah Fitur\nDibuang"
]
for c, h in enumerate(headers4, 1):
    cell = ws4.cell(row=2, column=c, value=h)
    style_cell(cell, fill=SUBHDR_FILL, font=SUBHDR_FONT)

col_widths4 = [14, 10, 10, 18, 22, 16, 18, 16, 16]
for c, w in enumerate(col_widths4, 1):
    ws4.column_dimensions[get_column_letter(c)].width = w

for r, sn in enumerate(["A","B","C","D","E","F","G","H"], 3):
    ws4.row_dimensions[r].height = 20
    bp = best_params[sn]
    opt_best = scenario_data[sn]["opt_best"]
    last_row = opt_best.iloc[-1]
    best_iter_found = int(last_row["iteration"])
    best_fit_acc = last_row["accuracy"]
    elapsed_total = last_row["elapsed_time_sec"] / 60.0  # to minutes
    epoch_rt = last_row["epoch_runtime_sec"]

    final_df = scenario_data[sn]["final"].set_index("scenario")
    own_key = bp["eval_key"]
    n_feat = int(final_df.loc[own_key, "num_features"]) if own_key in final_df.index else None
    total_feat = 768
    n_disc = total_feat - n_feat if n_feat is not None else None

    fill = SCEN_FILLS[sn]
    row_data = [
        f"Skenario {sn}", bp["best_pop"], bp["best_iter"],
        best_iter_found, best_fit_acc,
        round(elapsed_total, 2), round(epoch_rt, 2),
        n_feat, n_disc,
    ]
    for c, val in enumerate(row_data, 1):
        cell = ws4.cell(row=r, column=c, value=val)
        style_cell(cell, fill=fill, font=NORMAL_FONT, border=THIN_BORDER)
        if c == 5:
            cell.number_format = "0.0000"
        if c in (6, 7):
            cell.number_format = "0.00"
        if c == 1:
            cell.font = Font(name="Calibri", bold=True, size=10)

ws4.freeze_panes = "B3"


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 5 : Matriks Akurasi (pivot: Fitness Scenario x Eval Scenario)
# ─────────────────────────────────────────────────────────────────────────────
ws5 = wb.create_sheet("Matriks Akurasi")
ws5.row_dimensions[1].height = 36
ws5.row_dimensions[2].height = 22
ws5.row_dimensions[3].height = 50

ws5.merge_cells("A1:J1")
tc5 = ws5.cell(row=1, column=1,
               value="MATRIKS AKURASI: FITNESS SKENARIO (baris) × EVALUASI SKENARIO (kolom)")
style_cell(tc5, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=13))

ws5.merge_cells("A2:J2")
sub5 = ws5.cell(row=2, column=1,
    value="Diagonal (★) = evaluasi yang sesuai dengan skenario fitness. Nilai disorot biru.")
style_cell(sub5, fill=SUBHDR_FILL, font=Font(name="Calibri", bold=False, color="FFFFFF", size=10))

ws5.cell(row=3, column=1, value="Fitness\nSkenario \\ Eval")
style_cell(ws5.cell(row=3, column=1), fill=SUBHDR_FILL, font=SUBHDR_FONT)
ws5.column_dimensions["A"].width = 20

for i, ek in enumerate(EVAL_KEYS_ORDER, 2):
    cell = ws5.cell(row=3, column=i, value=EVAL_LABELS[ek])
    style_cell(cell, fill=SCEN_FILLS[list(BEST.keys())[i-2]], font=Font(name="Calibri", bold=True, size=9))
    ws5.column_dimensions[get_column_letter(i)].width = 16

for r, sn in enumerate(["A","B","C","D","E","F","G","H"], 4):
    ws5.row_dimensions[r].height = 22
    ws5.cell(row=r, column=1, value=f"Skenario {sn}\n{SCENARIO_LABELS[sn][:35]}…")
    style_cell(ws5.cell(row=r, column=1), fill=SCEN_FILLS[sn],
               font=Font(name="Calibri", bold=True, size=9), align=LEFT)
    final_df = scenario_data[sn]["final"].set_index("scenario")
    own_key = best_params[sn]["eval_key"]
    for c, ek in enumerate(EVAL_KEYS_ORDER, 2):
        val = final_df.loc[ek, "mean_accuracy"] if ek in final_df.index else None
        cell = ws5.cell(row=r, column=c, value=val)
        if ek == own_key:
            style_cell(cell, fill=PatternFill("solid", fgColor="BDD7EE"),
                       font=Font(name="Calibri", bold=True, size=11, color="1F4E79"),
                       border=Border(left=thick_side, right=thick_side, top=thick_side, bottom=thick_side),
                       num_format="0.0000")
            cell.value = f"★ {val:.4f}" if val else val
        else:
            style_cell(cell, fill=PatternFill("solid", fgColor="F2F2F2"),
                       font=NORMAL_FONT, border=THIN_BORDER, num_format="0.0000")

# Color scale for entire matrix (B4:I11), skip the diagonal-styled cells
for col_idx in range(2, 10):
    cl = get_column_letter(col_idx)
    # We skip adding color scale over starred cells; just do it on all
ws5.conditional_formatting.add(
    "B4:I11",
    ColorScaleRule(
        start_type="min",  start_color="FFC7CE",
        mid_type="percentile", mid_value=50, mid_color="FFEB9C",
        end_type="max",    end_color="C6EFCE"
    )
)

ws5.freeze_panes = "B4"


# ─────────────────────────────────────────────────────────────────────────────
# SHEET 6 : Rangkuman terbaik
# ─────────────────────────────────────────────────────────────────────────────
ws6 = wb.create_sheet("Rangkuman Terbaik")
ws6.row_dimensions[1].height = 40
ws6.row_dimensions[2].height = 24

ws6.merge_cells("A1:F1")
tc6 = ws6.cell(row=1, column=1, value="RANGKUMAN: SKENARIO FITNESS TERBAIK")
style_cell(tc6, fill=HEADER_FILL, font=Font(name="Calibri", bold=True, color="FFFFFF", size=14))

# Best overall (highest mean_accuracy on own eval key)
best_sn = max(
    ["A","B","C","D","E","F","G","H"],
    key=lambda s: scenario_data[s]["final"].set_index("scenario").loc[
        best_params[s]["eval_key"], "mean_accuracy"
    ]
)

summary_data = []
for sn in ["A","B","C","D","E","F","G","H"]:
    own_key = best_params[sn]["eval_key"]
    final_df = scenario_data[sn]["final"].set_index("scenario")
    own_acc = final_df.loc[own_key, "mean_accuracy"]
    summary_data.append({
        "Fitness Skenario": f"Skenario {sn}",
        "Deskripsi Fitness": SCENARIO_LABELS[sn],
        "Eval Key": own_key,
        "Best Pop": BEST[sn]["pop"],
        "Best Iter": BEST[sn]["iter"],
        "Mean Accuracy (own eval)": own_acc,
        "Std Accuracy": final_df.loc[own_key, "std_accuracy"],
        "Jumlah Fitur": int(final_df.loc[own_key, "num_features"]),
        "Waktu (menit)": best_params[sn]["elapsed_mnt"],
        "Ranking": 0,  # filled below
    })

summary_data.sort(key=lambda x: -x["Mean Accuracy (own eval)"])
for i, d in enumerate(summary_data, 1):
    d["Ranking"] = i

headers6 = ["Ranking", "Fitness Skenario", "Deskripsi Fitness", "Best Pop", "Best Iter",
            "Mean Accuracy\n(own eval)", "Std Accuracy", "Jumlah Fitur", "Waktu (menit)"]
col_widths6 = [8, 16, 42, 10, 10, 20, 14, 14, 14]
for c, h in enumerate(headers6, 1):
    cell = ws6.cell(row=2, column=c, value=h)
    style_cell(cell, fill=SUBHDR_FILL, font=SUBHDR_FONT)
    ws6.column_dimensions[get_column_letter(c)].width = col_widths6[c-1]

for r, d in enumerate(summary_data, 3):
    ws6.row_dimensions[r].height = 22
    sn = d["Fitness Skenario"].split()[-1]
    fill = SCEN_FILLS[sn]
    row_vals = [d["Ranking"], d["Fitness Skenario"], d["Deskripsi Fitness"],
                d["Best Pop"], d["Best Iter"],
                d["Mean Accuracy (own eval)"], d["Std Accuracy"],
                d["Jumlah Fitur"], d["Waktu (menit)"]]
    for c, val in enumerate(row_vals, 1):
        cell = ws6.cell(row=r, column=c, value=val)
        style_cell(cell, fill=fill, font=NORMAL_FONT, border=THIN_BORDER)
        if c in (6, 7):
            cell.number_format = "0.0000"
        if c == 9:
            cell.number_format = "0.0"
        if c == 1:
            cell.font = Font(name="Calibri", bold=True, size=12)
    # Highlight rank 1
    if d["Ranking"] == 1:
        gold = PatternFill("solid", fgColor="FFD700")
        for c in range(1, 10):
            ws6.cell(row=r, column=c).fill = gold
            ws6.cell(row=r, column=c).font = Font(name="Calibri", bold=True, size=11)

# Add note
ws6.row_dimensions[12].height = 18
note_cell = ws6.cell(row=12, column=1,
    value=f"★ Skenario terbaik berdasarkan akurasi evaluasi sesuai fitness: Skenario {best_sn}")
note_cell.font = Font(name="Calibri", bold=True, size=11, color="1F4E79")
ws6.merge_cells("A12:F12")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "ANALISA_PERBANDINGAN_FITNESS_SKENARIO_A_H.xlsx")
wb.save(out_path)
print(f"Saved: {out_path}")
