import os
import re
import json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import Rule

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "HASIL_ANALISA_CLGWO_Scenario_A_H.xlsx")

SCENARIOS_GWO = list("ABCDEFGH")

# ── helpers ────────────────────────────────────────────────────────────────────

def parse_exp_folder_name(folder_name):
    """Extract gwo_scenario, pop_size, max_iter, timestamp from folder name."""
    m = re.search(
        r"cnn_CL_GWO_scen([A-H])_hybrid_pop(\d+)_iter(\d+)_pop\d+_iter\d+_(\d{8}_\d{6})",
        folder_name,
    )
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    return None, None, None, None


def thin_border():
    t = Side(style="thin")
    return Border(left=t, right=t, top=t, bottom=t)


# ── collect data ────────────────────────────────────────────────────────────────

phase1_rows = []
phase2_rows = []
final_rows = []
fold_rows = []
opt_best_rows = []

for gwo_scen in SCENARIOS_GWO:
    scen_dir = os.path.join(BASE_DIR, f"results_CLGWO_scen{gwo_scen}")
    if not os.path.isdir(scen_dir):
        continue

    # --- phase1 pop search ---
    p1 = os.path.join(scen_dir, f"phase1_pop_search_scen{gwo_scen}.csv")
    if os.path.isfile(p1):
        df = pd.read_csv(p1)
        df.insert(0, "gwo_scenario", gwo_scen)
        phase1_rows.append(df)

    # --- phase2 iter search ---
    p2 = os.path.join(scen_dir, f"phase2_iter_search_scen{gwo_scen}.csv")
    if os.path.isfile(p2):
        df = pd.read_csv(p2)
        df.insert(0, "gwo_scenario", gwo_scen)
        phase2_rows.append(df)

    # --- per-experiment folders ---
    for item in sorted(os.listdir(scen_dir)):
        exp_path = os.path.join(scen_dir, item)
        if not os.path.isdir(exp_path):
            continue
        gwo_s, pop, itr, ts = parse_exp_folder_name(item)
        if gwo_s is None:
            continue

        meta = dict(gwo_scenario=gwo_s, pop_size=pop, max_iter=itr, timestamp=ts,
                    exp_folder=item)

        # final classification
        f_final = os.path.join(exp_path, "final_classification_results.csv")
        if os.path.isfile(f_final):
            df = pd.read_csv(f_final)
            for k, v in meta.items():
                df.insert(0, k, v) if k not in df.columns else None
            final_rows.append(df)

        # fold classification
        f_fold = os.path.join(exp_path, "fold_classification_results.csv")
        if os.path.isfile(f_fold):
            df = pd.read_csv(f_fold)
            for k, v in meta.items():
                df.insert(0, k, v) if k not in df.columns else None
            fold_rows.append(df)

        # optimization history best (last row = best reached)
        f_opt = os.path.join(exp_path, "optimization_history_best.csv")
        if os.path.isfile(f_opt):
            try:
                df = pd.read_csv(f_opt)
                # keep only: iteration, accuracy, fitness, elapsed_time_sec, num_selected, num_discarded
                keep_cols = [c for c in ["iteration", "accuracy", "fitness",
                                          "elapsed_time_sec", "num_selected",
                                          "num_discarded"] if c in df.columns]
                df = df[keep_cols].tail(1).copy()
                for k, v in meta.items():
                    df.insert(0, k, v)
                opt_best_rows.append(df)
            except Exception:
                pass

# build DataFrames
df_phase1 = pd.concat(phase1_rows, ignore_index=True) if phase1_rows else pd.DataFrame()
df_phase2 = pd.concat(phase2_rows, ignore_index=True) if phase2_rows else pd.DataFrame()
df_final  = pd.concat(final_rows,  ignore_index=True) if final_rows  else pd.DataFrame()
df_fold   = pd.concat(fold_rows,   ignore_index=True) if fold_rows   else pd.DataFrame()
df_opt    = pd.concat(opt_best_rows, ignore_index=True) if opt_best_rows else pd.DataFrame()

# ── build pivot / summary tables ───────────────────────────────────────────────

# Best params per GWO scenario (highest mean_accuracy across all experiments)
best_params = (
    df_final.groupby(["gwo_scenario", "pop_size", "max_iter", "scenario"])["mean_accuracy"]
    .mean()
    .reset_index()
) if not df_final.empty else pd.DataFrame()

best_per_gwo = pd.DataFrame()
if not best_params.empty:
    # best pop+iter combination for each gwo_scenario (highest avg across all classification scenarios)
    avg_by_params = (
        df_final.groupby(["gwo_scenario", "pop_size", "max_iter"])["mean_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"mean_accuracy": "avg_accuracy_all_scenarios"})
    )
    idx = avg_by_params.groupby("gwo_scenario")["avg_accuracy_all_scenarios"].idxmax()
    best_per_gwo = avg_by_params.loc[idx].reset_index(drop=True)

# Pivot: final results — rows=GWO scenario+params, cols=classification scenario, value=mean_accuracy
pivot_final = pd.DataFrame()
if not df_final.empty:
    pivot_final = df_final.pivot_table(
        index=["gwo_scenario", "pop_size", "max_iter", "exp_folder"],
        columns="scenario",
        values="mean_accuracy",
        aggfunc="first",
    ).reset_index()
    pivot_final.columns.name = None

# Pivot: fold results — mean accuracy per fold per classification scenario
pivot_fold = pd.DataFrame()
if not df_fold.empty:
    pivot_fold = df_fold.pivot_table(
        index=["gwo_scenario", "pop_size", "max_iter", "scenario"],
        columns="fold",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    pivot_fold.columns.name = None
    pivot_fold.columns = [str(c) if isinstance(c, int) else c for c in pivot_fold.columns]
    fold_cols = [c for c in pivot_fold.columns if c.isdigit()]
    pivot_fold["mean_accuracy"] = pivot_fold[fold_cols].mean(axis=1)
    pivot_fold["std_accuracy"]  = pivot_fold[fold_cols].std(axis=1)

# ── write to Excel ──────────────────────────────────────────────────────────────

print("Writing Excel …")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    def write_sheet(df, sheet_name):
        if df.empty:
            df = pd.DataFrame({"(no data)": []})
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    write_sheet(df_phase1,    "1_Phase1_PopSearch")
    write_sheet(df_phase2,    "2_Phase2_IterSearch")
    write_sheet(best_per_gwo, "3_BestParams_PerGWOScen")
    write_sheet(pivot_final,  "4_FinalResults_Pivot")
    write_sheet(df_final,     "5_FinalResults_All")
    write_sheet(pivot_fold,   "6_FoldResults_Pivot")
    write_sheet(df_fold,      "7_FoldResults_All")
    write_sheet(df_opt,       "8_OptHistory_Best")

# ── styling ────────────────────────────────────────────────────────────────────

wb = load_workbook(OUTPUT_FILE)

HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT  = Font(bold=True, color="FFFFFF", size=11)
ALT_FILL     = PatternFill("solid", fgColor="D6E4F0")
BEST_FILL    = PatternFill("solid", fgColor="C6EFCE")
BEST_FONT    = Font(bold=True, color="276221")
BORDER       = thin_border()

def style_sheet(ws, freeze="A2", col_widths=None):
    # header row
    for cell in ws[1]:
        cell.fill  = HEADER_FILL
        cell.font  = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER

    # data rows — alternating fill + borders
    for i, row in enumerate(ws.iter_rows(min_row=2), start=2):
        fill = ALT_FILL if i % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        for cell in row:
            cell.fill   = fill
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # auto column width
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            except Exception:
                pass
        if col_widths and col_letter in col_widths:
            ws.column_dimensions[col_letter].width = col_widths[col_letter]
        else:
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)

    if freeze:
        ws.freeze_panes = freeze
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[1].height = 30

for sname in wb.sheetnames:
    style_sheet(wb[sname])

# highlight best (is_best = YES) in phase search sheets
for sname in ["1_Phase1_PopSearch", "2_Phase2_IterSearch"]:
    ws = wb[sname]
    headers = [c.value for c in ws[1]]
    is_best_col = None
    for idx, h in enumerate(headers, 1):
        if h and "is_best" in str(h).lower():
            is_best_col = idx
            break
    if is_best_col:
        for row in ws.iter_rows(min_row=2):
            if row[is_best_col - 1].value == "YES":
                for cell in row:
                    cell.fill = BEST_FILL
                    cell.font = BEST_FONT

# highlight best params sheet
ws_bp = wb["3_BestParams_PerGWOScen"]
if ws_bp.max_row > 1:
    headers = [c.value for c in ws_bp[1]]
    acc_col = None
    for idx, h in enumerate(headers, 1):
        if h and "accuracy" in str(h).lower():
            acc_col = idx
            break
    if acc_col:
        # add color scale on accuracy column
        col_letter = get_column_letter(acc_col)
        ws_bp.conditional_formatting.add(
            f"{col_letter}2:{col_letter}{ws_bp.max_row}",
            ColorScaleRule(
                start_type="min", start_color="FFEB9C",
                mid_type="percentile", mid_value=50, mid_color="FFEB9C",
                end_type="max", end_color="63BE7B",
            ),
        )

# color scale on accuracy columns in pivot final and fold pivot
for sname in ["4_FinalResults_Pivot", "6_FoldResults_Pivot", "5_FinalResults_All"]:
    ws = wb[sname]
    headers = [c.value for c in ws[1]]
    for idx, h in enumerate(headers, 1):
        if h and ("accuracy" in str(h).lower() or str(h).startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"))):
            col_letter = get_column_letter(idx)
            ws.conditional_formatting.add(
                f"{col_letter}2:{col_letter}{ws.max_row}",
                ColorScaleRule(
                    start_type="min",  start_color="FFEB9C",
                    end_type="max",    end_color="63BE7B",
                ),
            )

# tab colors
TAB_COLORS = {
    "1_Phase1_PopSearch":    "2196F3",
    "2_Phase2_IterSearch":   "2196F3",
    "3_BestParams_PerGWOScen": "4CAF50",
    "4_FinalResults_Pivot":  "FF9800",
    "5_FinalResults_All":    "FF9800",
    "6_FoldResults_Pivot":   "9C27B0",
    "7_FoldResults_All":     "9C27B0",
    "8_OptHistory_Best":     "F44336",
}
for sname, color in TAB_COLORS.items():
    if sname in wb.sheetnames:
        wb[sname].sheet_properties.tabColor = color

wb.save(OUTPUT_FILE)
print(f"Done → {OUTPUT_FILE}")
print(f"  Phase1 rows  : {len(df_phase1)}")
print(f"  Phase2 rows  : {len(df_phase2)}")
print(f"  Final rows   : {len(df_final)}")
print(f"  Fold rows    : {len(df_fold)}")
print(f"  OptBest rows : {len(df_opt)}")
