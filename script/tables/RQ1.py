import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# -----------------------------
# Load data
# -----------------------------
csv = "report_RQ1.csv"
df = pd.read_csv(csv)
metrics = ["cpu", "mem",  "socket", "delay", "loss", "disk"]

# Aggregate by mean
# drop baro with baro = true
df = df[~((df["method"] == "baro") & (df["with_baro"] == True))]
agg_df = df.groupby(["method", "dataset", "with_baro"])[metrics].mean().reset_index()

# Make with_baro readable
agg_df["with_baro"] = agg_df["with_baro"].astype(str)
agg_df["method_combo"] = agg_df["method"] + " | baro=" + agg_df["with_baro"]

# -----------------------------
# Prepare figure layout
# -----------------------------
datasets = agg_df["dataset"].unique()
n = len(datasets)
cols = 2  # number of columns of subplots
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = axes.flatten()

# -----------------------------
# Plot each dataset
# -----------------------------
# order datasets so that they appear as
"""
Re1 | RE2
RE1 | RE2
RE1 | RE2
"""
desired_order = ["re1-ob","re2-ob", "re1-tt","re2-tt","re1-ss",  "re2-ss"]  # put the datasets in the order you want
datasets = [d for d in desired_order if d in agg_df["dataset"].unique()]
for i, dataset in enumerate(datasets):
    dataset_df = agg_df[agg_df["dataset"] == dataset]

    heatmap_data = dataset_df.set_index("method_combo")[metrics]

    
    sns.heatmap(
        heatmap_data,
        annot=heatmap_data, fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Mean Value"},
        ax=axes[i]
    )


    axes[i].set_title(f"{dataset.upper()}", fontsize=14, weight="bold")
    axes[i].set_xlabel("Metric")
    axes[i].set_ylabel("Method | BARO Setting")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].tick_params(axis='y', rotation=0)
    #save each subplot as a separate pdf under formde ouptut/figures/RQ1_<dataset>.pdf
    # make the fig on its own
    fig = plt.figure()
    sns.heatmap(
        heatmap_data,
        annot=heatmap_data, fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Mean Value"},
    )
    plt.title(f"{dataset.upper()}", fontsize=14, weight="bold")
    plt.xlabel("Metric")
    plt.ylabel("Method | BARO Setting")
    plt.tick_params(axis='x', rotation=30)
    plt.tick_params(axis='y', rotation=0)
    plt.savefig(f"output/figures/RQ1_{dataset}.pdf", bbox_inches='tight')

    # print the contents of each subplot and save it to a text file
    with open(f"output/figures/RQ1_{dataset}.txt", "w") as f:
        f.write(f"Dataset: {dataset.upper()}\n")
        f.write(heatmap_data.to_string())   
        

plt.tight_layout()
plt.show()
