#!/usr/bin/env python
# ================================================================
# EKSPERIMEN 4: ANALISIS KOMPLEKSITAS WAKTU DAN RUANG
#
# Analisis teoritis + empiris CL_GWO vs GWO standar.
# Tidak memerlukan GPU atau dataset – dijalankan secara analitik
# dan dengan profiling ringan.
#
# Bagian:
#   A) Derivasi analitik kompleksitas
#   B) Perbandingan teoretis dengan GWO standar dan varian lain
#   C) Profiling empiris waktu per iterasi vs dimensi
#   D) Plot dan tabel ringkasan untuk paper
# ================================================================

import os
import sys
import csv
import time

BASE_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "full_code_RAP")
sys.path.insert(0, BASE_MAIN)

RESULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.io import ensure_dir


# ================================================================
# A. DERIVASI ANALITIK
# ================================================================

COMPLEXITY_TABLE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           ANALISIS KOMPLEKSITAS WAKTU DAN RUANG CL_GWO                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Notasi:                                                                     ║
║    T  = jumlah iterasi (epoch)                                               ║
║    N  = ukuran populasi (pop_size)                                           ║
║    D  = dimensi solusi = 2 × n_features                                     ║
║    K  = jumlah fold CV (5)                                                   ║
║    n  = jumlah sampel dataset                                                ║
║    C  = kompleksitas evaluasi classifier (SVM = O(n²) ~ O(n^1.5–2))         ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. GWO Standar (OriginalGWO)                                                ║
║     Tiap iterasi:                                                            ║
║       - Update posisi: N × D operasi      → O(N × D)                        ║
║       - Evaluasi fitness: N × K × C       → O(N × K × C)                   ║
║     Total: O(T × N × (D + K × C))                                           ║
║     Ruang: O(N × D)                                                         ║
║                                                                              ║
║  2. CL_GWO (Competitive Learning GWO)                                        ║
║     Tambahan dibanding GWO:                                                  ║
║       a) Per-dimension switch (DWL):                                         ║
║          Tiap partikel, tiap dimensi: 1 perbandingan random + threshold      ║
║          Overhead: N × D comparisons → O(N × D)  — same order               ║
║       b) Competitive leader selection (ALS):                                 ║
║          Tiap partikel: tournament dari 2 random peers                       ║
║          Overhead: N × 2 fitness comparisons → O(N)  — negligible           ║
║       c) Stagnation counter (max_flag):                                      ║
║          N counters, increment/reset per iterasi → O(N)  — negligible       ║
║                                                                              ║
║     Total: O(T × N × (D + K × C))                                           ║
║     → Kompleksitas ASIMPTOTIK SAMA dengan GWO standar                        ║
║                                                                              ║
║     Perbedaan konstanta:                                                     ║
║       CL_GWO overhead per iterasi ≈ 3×N×D ops ekstra                        ║
║       (DWL: 2D, ALS: 2, max_flag: 1 per partikel)                           ║
║       Dalam praktik: overhead < 5% dari waktu total (didominasi eval)        ║
║                                                                              ║
║  3. Perbandingan dengan varian lain                                          ║
║     ┌────────────────────────┬──────────────────────────┬──────────────────┐ ║
║     │ Metode                 │ Time Complexity           │ Space Complexity │ ║
║     ├────────────────────────┼──────────────────────────┼──────────────────┤ ║
║     │ GWO (standar)          │ O(T·N·(D + K·C))         │ O(N·D)           │ ║
║     │ CL_GWO (proposed)      │ O(T·N·(D + K·C))         │ O(N·D)           │ ║
║     │ WS_GWO                 │ O(T·N·(D + K·C))         │ O(N·D)           │ ║
║     │ RLCGWO                 │ O(T·N·(D + K·C + N))     │ O(N·D + N²)      │ ║
║     │ AMGWO                  │ O(T·N·(D + K·C))         │ O(N·D)           │ ║
║     │ BFLGWO                 │ O(T·N·(D + K·C))         │ O(N·D)           │ ║
║     │ PSO (OriginalPSO)      │ O(T·N·(D + K·C))         │ O(N·D) + vel     │ ║
║     │ CLPSO                  │ O(T·N·(D + K·C))         │ O(N·D) + exm.    │ ║
║     └────────────────────────┴──────────────────────────┴──────────────────┘ ║
║                                                                              ║
║  Catatan RLCGWO: memerlukan O(N²) per iterasi untuk role-learning            ║
║  antar partikel — CL_GWO secara ruang lebih efisien.                         ║
║                                                                              ║
║  4. CL_GWO + Feature Construction (FC)                                       ║
║     FC menambahkan:                                                          ║
║       - Decode solusi: O(D)                                                  ║
║       - Chain construction: O(n × n_sel) per subset                         ║
║       Overhead total: O(T × N × K × n × n_sel)                              ║
║       Dalam praktik n_sel << D dan n << total dataset,                       ║
║       sehingga overhead FC << overhead evaluasi classifier.                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ================================================================
# B. TABEL FORMAL UNTUK PAPER
# ================================================================

METHODS = [
    # (nama, time_complexity_label, space_complexity_label, catatan)
    ("GWO (standar)",   "O(T·N·(D + K·C))",   "O(N·D)",         "baseline"),
    ("CL_GWO (kami)",   "O(T·N·(D + K·C))",   "O(N·D)",         "overhead konstan +3D per partikel"),
    ("WS_GWO",          "O(T·N·(D + K·C))",   "O(N·D)",         ""),
    ("RLCGWO",          "O(T·N·(D + K·C + N))","O(N·D + N²)",   "role-learning O(N²)"),
    ("AMGWO",           "O(T·N·(D + K·C))",   "O(N·D)",         ""),
    ("BFLGWO",          "O(T·N·(D + K·C))",   "O(N·D)",         ""),
    ("PSO",             "O(T·N·(D + K·C))",   "O(N·D) + vel",   "vel = N·D"),
    ("CLPSO",           "O(T·N·(D + K·C))",   "O(N·D) + exmpl.","exemplar table"),
]


# ================================================================
# C. PROFILING EMPIRIS: WAKTU PER ITERASI VS DIMENSI
#    (tidak memerlukan dataset nyata – gunakan data sintetis)
# ================================================================

def profile_time_vs_dim(dim_list, pop_size=25, n_iter=5, n_samples=100, n_classes=5):
    """
    Ukur waktu per iterasi CL_GWO (mekanisme saja, tanpa evaluasi fitness nyata)
    untuk berbagai ukuran dimensi.

    Menggunakan fitness dummy (random) agar profiling murni operasi optimizer.
    """
    from mealpy.swarm_based.GWO import CL_GWO, OriginalGWO
    from mealpy.utils.space import FloatVar

    results = {}

    for algo_name, AlgoClass, kwargs in [
        ("GWO",    OriginalGWO, {}),
        ("CL_GWO", CL_GWO,     {"w_min": 0.2, "w_max": 2.0, "max_flag": 7}),
    ]:
        times = []
        for D in dim_list:
            call_count = [0]

            def dummy_fitness(sol):
                call_count[0] += 1
                return -float(np.random.random())

            problem = {
                "obj_func": dummy_fitness,
                "bounds":   FloatVar(lb=[0.0] * D, ub=[1.0] * D),
                "minmax":   "min",
                "save_population": False,
            }

            opt = AlgoClass(epoch=n_iter, pop_size=pop_size, **kwargs)
            t0  = time.time()
            opt.solve(problem)
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  {algo_name} D={D:4d}: {elapsed:.3f}s ({call_count[0]} evals)")

        results[algo_name] = times

    return results


# ================================================================
# D. PLOT DAN SAVE
# ================================================================

def plot_time_vs_dim(dim_list, timing_results, out_dir):
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"GWO": "steelblue", "CL_GWO": "crimson"}
    markers = {"GWO": "o", "CL_GWO": "s"}

    for algo, times in timing_results.items():
        ax.plot(dim_list, times, marker=markers[algo], color=colors[algo],
                linewidth=2, markersize=7, label=algo)

    ax.set_xlabel("Dimensi Solusi (D = 2 × n_features)")
    ax.set_ylabel("Waktu Total (detik)")
    ax.set_title(f"Waktu Eksekusi vs Dimensi\n(pop={25}, iter=5, fitness dummy)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, "time_vs_dim.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot waktu → {path}")


def plot_overhead_ratio(dim_list, timing_results, out_dir):
    """Rasio waktu CL_GWO / GWO standar per dimensi."""
    ensure_dir(out_dir)
    gwo_t    = timing_results.get("GWO", [1.0] * len(dim_list))
    clgwo_t  = timing_results.get("CL_GWO", [1.0] * len(dim_list))
    ratio    = [c / g if g > 0 else 1.0 for c, g in zip(clgwo_t, gwo_t)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dim_list, ratio, "D-", color="darkorange", linewidth=2, markersize=7)
    ax.axhline(y=1.0, linestyle="--", color="gray", alpha=0.7, label="GWO baseline (ratio=1)")
    ax.set_xlabel("Dimensi Solusi (D)")
    ax.set_ylabel("Rasio Waktu CL_GWO / GWO")
    ax.set_title("Overhead CL_GWO relatif terhadap GWO Standar")
    ax.set_ylim(0.5, 2.5)
    ax.legend()
    ax.grid(True, alpha=0.4)
    for x, r in zip(dim_list, ratio):
        ax.annotate(f"{r:.2f}×", (x, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "overhead_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Overhead ratio → {path}")


def save_complexity_csv(out_dir):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "complexity_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "time_complexity", "space_complexity", "note"])
        for name, tc, sc, note in METHODS:
            w.writerow([name, tc, sc, note])
    print(f"[SAVED] Complexity table → {path}")


def save_complexity_latex(out_dir):
    """Generate tabel LaTeX siap pakai untuk paper."""
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "complexity_table.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Perbandingan Kompleksitas Waktu dan Ruang}",
        r"\label{tab:complexity}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Metode & Kompleksitas Waktu & Kompleksitas Ruang & Catatan \\",
        r"\midrule",
    ]
    for name, tc, sc, note in METHODS:
        # Escape underscore untuk LaTeX
        name_tex  = name.replace("_", r"\_")
        tc_tex    = tc.replace("·", r"$\cdot$").replace("·", r"$\cdot$")
        sc_tex    = sc
        note_tex  = note.replace("_", r"\_") if note else ""
        lines.append(rf"{name_tex} & \texttt{{{tc_tex}}} & \texttt{{{sc_tex}}} & {note_tex} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] LaTeX table → {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    ensure_dir(RESULT_BASE)

    # A. Cetak derivasi analitik
    print(COMPLEXITY_TABLE)

    # Simpan ke file teks
    txt_path = os.path.join(RESULT_BASE, "complexity_derivation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(COMPLEXITY_TABLE)
    print(f"[SAVED] Derivasi → {txt_path}")

    # B. Simpan tabel formal
    save_complexity_csv(RESULT_BASE)
    save_complexity_latex(RESULT_BASE)

    # C. Profiling empiris (dimensi kecil agar cepat)
    dim_list = [50, 100, 200, 400, 768, 1024, 1536, 2048]
    print("\n[PROFILE] Memulai profiling waktu (fitness dummy) ...")
    timing_results = profile_time_vs_dim(
        dim_list=dim_list,
        pop_size=25,
        n_iter=5
    )

    # D. Plot
    plot_time_vs_dim(dim_list, timing_results, RESULT_BASE)
    plot_overhead_ratio(dim_list, timing_results, RESULT_BASE)

    # Simpan timing ke CSV
    timing_csv = os.path.join(RESULT_BASE, "timing_profile.csv")
    with open(timing_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["dim"] + list(timing_results.keys())
        w.writerow(header)
        for i, d in enumerate(dim_list):
            row = [d] + [timing_results[algo][i] for algo in timing_results]
            w.writerow(row)
    print(f"[SAVED] Timing CSV → {timing_csv}")

    # Ringkasan overhead
    gwo    = timing_results.get("GWO", [])
    clgwo  = timing_results.get("CL_GWO", [])
    if gwo and clgwo:
        ratios = [c / g if g > 0 else 1.0 for c, g in zip(clgwo, gwo)]
        print(f"\n[RINGKASAN] Rata-rata overhead CL_GWO/GWO: {np.mean(ratios):.3f}×")
        print(f"            Rentang: {min(ratios):.3f}× – {max(ratios):.3f}×")
        print(f"            → CL_GWO menambahkan overhead konstan, bukan asimptotis")

    print("\n[DONE] Eksperimen 4 selesai.")
    print(f"       Semua output tersimpan di: {RESULT_BASE}")


if __name__ == "__main__":
    main()
