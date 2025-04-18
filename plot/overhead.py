import matplotlib.pyplot as plt
import numpy as np

methods = ["PPO", "TPPO", "DDPG"]

action_cuda = [
    0.0011334214359521866,
    0.0016002077609300613,
    0.0006547588855028152
]

action_cpu = [
    0.00042392686009407043,
    0.0008370000869035721,
    0.0006539300084114075
]

x = np.arange(len(methods))  
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

bars_cuda = ax.bar(x - width/2, action_cuda, width, label='CUDA', color='C0')
bars_cpu = ax.bar(x + width/2, action_cpu, width, label='CPU', color='C1')

for bars in [bars_cuda, bars_cpu]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2e}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

ax.set_xlabel("Models")
ax.set_ylabel("Model Inference Time (seconds)")
ax.set_title("Inference Time")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.tight_layout()
plt.savefig("overhead")
