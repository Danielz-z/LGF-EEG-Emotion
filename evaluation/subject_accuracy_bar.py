import numpy as np
import matplotlib.pyplot as plt

accuracies = [
    0.4125, 0.3500, 0.3500, 0.2875, 0.4000,
    0.3500, 0.4416, 0.4500, 0.4750, 0.4250,
    0.4875, 0.4000, 0.3250, 0.4875, 0.4103,
    0.3625, 0.3590, 0.4625, 0.3250, 0.4500,
]

subjects = np.arange(1, len(accuracies) + 1)
labels = [f"S{i}" for i in subjects]

acc_arr = np.array(accuracies)
mean_acc = acc_arr.mean()

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(11.5, 4.8))

colors = ["#4C72B0" if acc >= mean_acc else "#DD8452" for acc in acc_arr]

ax.bar(
    subjects,
    acc_arr,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)

ax.axhline(
    mean_acc,
    linestyle="--",
    linewidth=1.4,
    color="gray",
    alpha=0.9,
    label=f"Mean = {mean_acc:.3f}"
)

ax.set_xticks(subjects)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Subject ID")

ax.set_ylim(0.25, 0.55)

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
ax.set_axisbelow(True)

ax.legend(frameon=False, loc="upper left")

ax.set_title(
    "Subject-wise Accuracy under LOSO Evaluation",
    pad=12
)

plt.tight_layout()
plt.savefig("subject_accuracy_bar.png", dpi=300)
plt.show()
