import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "/Users/virat_garg/Documents/Sem 5/ANLP/project/qwen_testing/conceptor_antonym/capitalise_final5/Qwen2.5-1.5B-Instruct_capitalize_conceptor_paper_2.csv"       # Change this to your CSV file
OUTPUT_DIR = Path("heatmaps_conceptor_paper")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# Remove "Average" rows
# ============================================================

# We only want actual experiment rows.
df = df[df["experiment"].astype(str) != "Average"].copy()


# Convert relevant columns to numeric
numeric_columns = [
    "layer",
    "beta",
    "aperture",
    "lambda_",
    "final_accuracy"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove rows with missing values
df = df.dropna(subset=numeric_columns)

print("After removing Average rows:", df.shape)


# ============================================================
# Function to create heatmap
# ============================================================

def create_heatmap(df, variable, output_name):
    """
    X-axis:
        variable

    Y-axis:
        layer

    Cell:
        Maximum final_accuracy over all other configurations.

    Example for variable = beta:

        max over aperture and lambda_

        H[layer, beta] =
            max(final_accuracy)
    """

    # --------------------------------------------------------
    # Take MAX over all other configurations
    # --------------------------------------------------------

    grouped = (
        df.groupby(["layer", variable])["final_accuracy"]
        .max()
        .reset_index()
    )

    # Convert into matrix
    heatmap_data = grouped.pivot(
        index="layer",
        columns=variable,
        values="final_accuracy"
    )

    # Sort rows and columns
    heatmap_data = heatmap_data.sort_index()
    heatmap_data = heatmap_data.reindex(
        sorted(heatmap_data.columns),
        axis=1
    )

    # --------------------------------------------------------
    # Determine color scale from ACTUAL data
    # --------------------------------------------------------

    max_value = heatmap_data.max().max()

    print(f"\n{variable} heatmap")
    print("Maximum value:", max_value)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        heatmap_data,

        # Show value inside every cell
        annot=True,

        # One decimal / two decimals depending on magnitude
        fmt=".2f",

        # Red -> Yellow -> Green
        cmap="RdYlGn",

        # IMPORTANT:
        # Don't force vmax=1.
        # Use actual maximum value.
        vmin=0,
        vmax=max_value,

        # Grid between cells
        linewidths=0.5,
        linecolor="white",

        # Colorbar
        cbar_kws={
            "label": "Maximum Final Accuracy"
        }
    )

    plt.xlabel(variable, fontsize=13)
    plt.ylabel("Layer", fontsize=13)

    plt.title(
        f"Maximum Final Accuracy vs Layer and {variable}",
        fontsize=15
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / output_name

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved → {output_path}")


# ============================================================
# Generate heatmaps
# ============================================================

create_heatmap(
    df,
    variable="beta",
    output_name="layer_vs_beta_max.png"
)

create_heatmap(
    df,
    variable="aperture",
    output_name="layer_vs_aperture_max.png"
)

create_heatmap(
    df,
    variable="lambda_",
    output_name="layer_vs_lambda_max.png"
)

print("\nAll heatmaps generated!")