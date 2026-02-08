import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def graph1(stream_csv: str, out_png: str):
    df = pd.read_csv(stream_csv)

    x = df["items_processed"]
    plt.figure(figsize=(8, 5))
    plt.plot(x, df["true_F0"], label="F0^t (истинное)", marker="o")
    plt.plot(x, df["hll_estimate_base"], label="N_t (baseline)", marker="x")
    plt.plot(x, df["hll_estimate_packed6"], label="N_t (packed6)", marker="^")

    plt.xlabel("Число обработанных элементов (t)")
    plt.ylabel("Количество уникальных")
    plt.title("График №1 (Stage 4): F0^t vs N_t (baseline vs packed6)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

def graph2(all_csv: str, out_png: str):
    df = pd.read_csv(all_csv)
    g = df.groupby("step_index")

    t = g["items_processed"].mean().values

    stats_b = g["hll_estimate_base"].agg(mean="mean", std="std").reset_index()
    stats_p = g["hll_estimate_packed6"].agg(mean="mean", std="std").reset_index()

    plt.figure(figsize=(8, 5))

    plt.plot(t, stats_b["mean"], label="E(N_t) baseline")
    plt.fill_between(t, stats_b["mean"] - stats_b["std"], stats_b["mean"] + stats_b["std"],
                     alpha=0.15, label="E(N_t) ± σ baseline")

    plt.plot(t, stats_p["mean"], label="E(N_t) packed6")
    plt.fill_between(t, stats_p["mean"] - stats_p["std"], stats_p["mean"] + stats_p["std"],
                     alpha=0.15, label="E(N_t) ± σ packed6")

    plt.xlabel("Число обработанных элементов (t)")
    plt.ylabel("Оценка количества уникальных")
    plt.title("График №2 (Stage 4): статистика оценки (baseline vs packed6)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

def print_errors(all_csv: str):
    df = pd.read_csv(all_csv)
    err_base = (df["hll_estimate_base"] - df["true_F0"]) / df["true_F0"]
    err_pack = (df["hll_estimate_packed6"] - df["true_F0"]) / df["true_F0"]

    print(f"Stage4 baseline: mean(err) = {err_base.mean():.12f}, std(err) = {err_base.std():.12f}")
    print(f"Stage4 packed6 : mean(err) = {err_pack.mean():.12f}, std(err) = {err_pack.std():.12f}")

    max_diff = (df["hll_estimate_base"] - df["hll_estimate_packed6"]).abs().max()
    print(f"Check: max |base - packed6| = {max_diff:.12f}")

def main():
    stream_csv = "stage4_hll_stream_0.csv"
    all_csv = "stage4_hll_results_all_streams.csv"

    if not Path(stream_csv).exists():
        raise FileNotFoundError(f"Not found: {stream_csv}")
    if not Path(all_csv).exists():
        raise FileNotFoundError(f"Not found: {all_csv}")

    graph1(stream_csv, "graph1_stage4.png")
    graph2(all_csv, "graph2_stage4.png")
    print_errors(all_csv)

if __name__ == "__main__":
    main()