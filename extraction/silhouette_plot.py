import re
import matplotlib.pyplot as plt

# Path to your file
file_path = "silhouette.txt"

layers = []
silhouette_scores = []

# Read and parse the file
with open(file_path, "r") as f:
    for line in f:
        match = re.search(
            r"layer\s+(\d+)\s+silhouette\s+([+-]?\d*\.?\d+)",
            line
        )

        if match:
            layer = int(match.group(1))
            score = float(match.group(2))

            layers.append(layer)
            silhouette_scores.append(score)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    layers,
    silhouette_scores,
    marker="o",
    linewidth=2
)

plt.xlabel("Layer", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.title("Silhouette Score vs. Layer", fontsize=14)

plt.xticks(layers)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("silhouette_vs_layer.png", dpi=300, bbox_inches="tight")
plt.show()