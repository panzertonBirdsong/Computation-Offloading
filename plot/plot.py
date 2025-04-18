import matplotlib.pyplot as plt
import numpy as np

# Example data: each dataset has 5 scores
data = {
    "Random Guessing": [
        4276.840642789202,
        3222.7546472437525,
        2263.470285609328,
        4070.240969553734,
        2221.2770210605245
    ],
    "PPO": [
        4911.86460624995,
        4284.783942712311,
        5169.725277644191,
        4737.581127495921,
        4840.150604735473
    ],
    "TPPO": [
        4502.023556280681,
        4686.654030560653,
        4740.967452912375,
        5522.7754983724435,
        5791.910230338759
    ],
    "DDPG": [
        5274.748456138321,
        5297.99345533114,
        4806.586944653572,
        5080.328339497522,
        5362.011570816294
    ]
}

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(data),
    figsize=(4 * len(data), 5),
    sharey=True
)

if len(data) == 1:
    axes = [axes]

xticks = np.arange(6)
xtick_labels = ["1", "2", "3", "4", "5", "avg"]

for ax, (dataset_name, scores) in zip(axes, data.items()):
    avg = sum(scores) / len(scores)
    plot_values = scores + [avg]
    
    bars = ax.bar(xticks, plot_values, color="C0")
    bars[-1].set_color("C3")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )


    ax.set_title(dataset_name)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Eval & Avg")
    ax.set_ylabel("Score")

fig.suptitle("Evaluation Results", fontsize=14)
plt.tight_layout()
plt.savefig("eva")
