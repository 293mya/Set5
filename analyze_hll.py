import pandas as pd
import matplotlib.pyplot as plt

stream_id = 0  
df_stream = pd.read_csv(f"hll_stream_{stream_id}.csv")

plt.figure(figsize=(8, 5))
plt.plot(df_stream["items_processed"], df_stream["true_F0"], label="F0^t (истинное)", marker="o")
plt.plot(df_stream["items_processed"], df_stream["hll_estimate"], label="N_t (HyperLogLog)", marker="x")
plt.xlabel("Число обработанных элементов (t)")
plt.ylabel("Количество уникальных")
plt.title(f"Сравнение F0^t и N_t для потока {stream_id}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df_all = pd.read_csv("hll_results_all_streams.csv")

grouped = df_all.groupby("step_index")

stats = grouped["hll_estimate"].agg(
    mean_est="mean",
    std_est="std"
).reset_index()
stats["items_processed"] = grouped["items_processed"].mean().values

plt.figure(figsize=(8, 5))
plt.plot(stats["items_processed"], stats["mean_est"], label="E(N_t)")
plt.fill_between(
    stats["items_processed"],
    stats["mean_est"] - stats["std_est"],
    stats["mean_est"] + stats["std_est"],
    color="C0", alpha=0.2, label="E(N_t) ± σ_{N_t}"
)
plt.xlabel("Число обработанных элементов (t)")
plt.ylabel("Оценка количества уникальных")
plt.title("Статистика оценки HyperLogLog")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_all["rel_error"] = (df_all["hll_estimate"] - df_all["true_F0"]) / df_all["true_F0"]
print("Средняя относительная ошибка:", df_all["rel_error"].mean())
print("Стандартное отклонение относительной ошибки:", df_all["rel_error"].std())