import os
import csv
import json
import pandas as pd
import sys
import platform
import torch


class UnifiedLogger:
    """
    Unified experiment logger.

    Menyimpan:
    1) Riwayat optimasi metaheuristik (global best + runtime asli)
    2) Solusi terbaik
    3) Hasil klasifikasi akhir
    4) Konfigurasi eksperimen (cfg)
    5) Informasi environment (Python, Torch, GPU)

    Seluruh artefak disimpan dalam satu folder eksperimen
    untuk menjamin reproducibility & reviewer-safety.
    """

    def __init__(self, save_dir, constructor, cfg=None):
        self.constructor = constructor
        self.save_dir = save_dir
        self.cfg = cfg

        # ==================================================
        # SIMPAN KONFIGURASI & ENVIRONMENT
        # ==================================================
        if self.cfg is not None:
            self._save_experiment_config()
            self._save_environment_info()

        # ==================================================
        # OPTIMIZATION HISTORY (GLOBAL BEST)
        # ==================================================
        self.path = os.path.join(save_dir, "optimization_history.csv")

        self.header = [
            "iteration",
            "accuracy",
            "fitness",
            "elapsed_time_sec",     # ⏱️ kumulatif (ASLI dari Mealpy)
            "epoch_runtime_sec",    # ⏱️ runtime epoch (ASLI dari Mealpy)

            "f_selected_baru",
            "f_discarded_baru",

            "num_selected",
            "num_discarded",

            "selected_features",
            "discarded_features",

            "operators_selected",
            "operators_discarded",

            "f_selected",
            "f_discarded",
        ]

        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    # ==================================================
    # SIMPAN KONFIGURASI EKSPERIMEN
    # ==================================================
    def _save_experiment_config(self):
        path = os.path.join(self.save_dir, "experiment_config.json")
        with open(path, "w") as f:
            json.dump(self.cfg, f, indent=4)
        print("[LOGGER] experiment_config.json disimpan")

    # ==================================================
    # SIMPAN INFORMASI ENVIRONMENT
    # ==================================================
    def _save_environment_info(self):
        path = os.path.join(self.save_dir, "environment.txt")
        with open(path, "w") as f:
            f.write(f"Python version : {sys.version}\n")
            f.write(f"Platform       : {platform.platform()}\n")
            f.write(f"Torch version  : {torch.__version__}\n")
            f.write(f"CUDA available : {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"GPU name       : {torch.cuda.get_device_name(0)}\n")
        print("[LOGGER] environment.txt disimpan")

    # ==================================================
    # LOG RIWAYAT GLOBAL BEST (+ RUNTIME ASLI MEALPY)
    # ==================================================
    def log_from_history(self, optimizer):
        """
        Menyimpan riwayat global best metaheuristik
        dengan runtime ASLI dari Mealpy (epoch-level).
        """

        X = optimizer.problem.obj_func.X
        history = optimizer.history.list_global_best

        # Runtime asli dari Mealpy (detik per epoch)
        # Sesuai dengan log: "Runtime: xxx seconds"
        epoch_times = optimizer.history.list_epoch_time

        cumulative_time = 0.0

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)

            for it, agent in enumerate(history):

                if agent is None or agent.target is None:
                    continue

                epoch_runtime = float(epoch_times[it])
                cumulative_time += epoch_runtime

                decoded = self.constructor.decode(agent.solution)
                if decoded is None:
                    continue

                f_sel, f_dis = self.constructor.construct_features_only(X, decoded)

                writer.writerow([
                    it,
                    -agent.target.fitness,          # accuracy
                    agent.target.fitness,           # fitness (minimization)
                    f"{cumulative_time:.6f}",       # ⏱️ kumulatif VALID
                    f"{epoch_runtime:.6f}",         # ⏱️ epoch VALID

                    float(f_sel[-1]),
                    float(f_dis[-1]),

                    len(decoded["selected"]),
                    len(decoded["discarded"]),

                    "|".join(map(str, decoded["selected"])),
                    "|".join(map(str, decoded["discarded"])),

                    "|".join(map(str, decoded["operators_selected"])),
                    "|".join(map(str, decoded["operators_discarded"])),

                    "|".join(f"{v:.6f}" for v in f_sel),
                    "|".join(f"{v:.6f}" for v in f_dis),
                ])

        print("[LOGGER] optimization_history.csv berhasil disimpan")

    # ==================================================
    # SIMPAN SOLUSI TERBAIK
    # ==================================================
    def save_best_only(self):
        if not os.path.exists(self.path):
            return

        df = pd.read_csv(self.path)
        df = df.dropna(subset=["accuracy"])
        if df.empty:
            return

        best = df.loc[df["accuracy"].astype(float).idxmax()]
        best_path = self.path.replace(".csv", "_best.csv")
        best.to_frame().T.to_csv(best_path, index=False)

        print("[LOGGER] Best solution disimpan:", best_path)

    # ==================================================
    # SIMPAN HASIL KLASIFIKASI AKHIR
    # ==================================================
    def save_final_classification_results(self, results_dict):
        path = os.path.join(self.save_dir, "final_classification_results.csv")

        header = [
            "scenario",
            "mean_accuracy",
            "std_accuracy",
            "num_features",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for scenario, res in results_dict.items():
                writer.writerow([
                    scenario,
                    f"{float(res['mean_acc']):.6f}",
                    f"{float(res['std_acc']):.6f}",
                    int(res["n_features"]),
                ])

        print("[LOGGER] final_classification_results.csv berhasil disimpan")
        print(" -", path)

        # Simpan detail per fold jika tersedia
        if any("fold_accs" in res for res in results_dict.values()):
            self._save_fold_results(results_dict)

    def _save_fold_results(self, results_dict):
        path = os.path.join(self.save_dir, "fold_classification_results.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "fold", "accuracy"])

            for scenario, res in results_dict.items():
                for fold_idx, acc in enumerate(res.get("fold_accs", []), start=1):
                    writer.writerow([
                        scenario,
                        fold_idx,
                        f"{float(acc):.6f}",
                    ])

        print("[LOGGER] fold_classification_results.csv berhasil disimpan")
        print(" -", path)

    # ==================================================
    # RIWAYAT POPULASI (NETRAL METODE, TANPA WAKTU)
    # ==================================================
    def save_population_history(self, optimizer):
        """
        Menyimpan riwayat populasi per epoch.
        NETRAL metode (PSO / GWO / GA / DE).
        TANPA runtime (menghindari klaim palsu).
        """

        path = os.path.join(self.save_dir, "population_history.csv")
        method_name = self.cfg.get("META_ALGO", "UNKNOWN")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "method",
                "epoch",
                "agent_id",
                "fitness",
                "accuracy",
                "num_selected",
                "num_discarded",
                "sparsity",
                "selected_features",
                "discarded_features",
            ])

            history = optimizer.history

            for epoch, population in enumerate(history.list_population):

                for pid, agent in enumerate(population):

                    if agent is None or agent.target is None:
                        continue

                    decoded = self.constructor.decode(agent.solution)
                    if decoded is None:
                        continue

                    num_selected = len(decoded["selected"])
                    num_discarded = len(decoded["discarded"])
                    sparsity = num_selected / self.constructor.n

                    writer.writerow([
                        method_name,
                        epoch + 1,
                        pid,
                        agent.target.fitness,
                        -agent.target.fitness,
                        num_selected,
                        num_discarded,
                        sparsity,
                        "|".join(map(str, decoded["selected"])),
                        "|".join(map(str, decoded["discarded"])),
                    ])

        print("[LOGGER] population_history.csv berhasil disimpan")
        print(" -", path)
