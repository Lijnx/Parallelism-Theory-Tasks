import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dgemv_time.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()

sizes = sorted(df["data_size"].unique())

plt.figure(figsize=(9,5))

# линия идеального ускорения
plt.plot(
    df["threads"],
    df["threads"],
    "--",
    linewidth=2,
    color="gray",
    label="Linear"
)

# линии для разных M
for size in sizes:
    part = df[df["data_size"] == size].sort_values("threads")

    plt.plot(
        part["threads"],
        part["acceleration"],
        marker="o",
        linewidth=2,
        label=f"M = {size}"
    )

plt.xlabel("p")
plt.ylabel(r"$S_p$")
plt.grid(True, alpha=0.3)

# легенда
plt.legend()

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
# plt.show()