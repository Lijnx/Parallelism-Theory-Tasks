import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("integrate_time.csv", skipinitialspace=True)

df = df.sort_values("threads")

plt.figure(figsize=(8,5))

# линия идеального ускорени
plt.plot(
    df["threads"],
    df["threads"],
    "--",
    linewidth=2,
    color="gray"
)

# подпись "Linear"
x_last = df["threads"].iloc[-1]
y_last = df["threads"].iloc[-1]

plt.text(
    x_last * 0.95,
    y_last * 0.95,
    "Linear",
    fontsize=11,
    ha="right"
)

plt.plot(
    df["threads"],
    df["acceleration"],
    marker="o",
    linewidth=2,
    label=""
)

plt.xlabel("p")
plt.ylabel(r"$S_p$")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
# plt.show()
