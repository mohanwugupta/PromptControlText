import matplotlib
matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse

def plot_safety_boundary(csv_path: str, output_path: str):
    """
    Reads the output from Phase 1 and plots Harmful Compliance vs False Refusal
    across prompt families. 
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Harmful compliance = Compliance on HarmBench 
    # False Refusal = Refusal on XSTest (safe prompts)
    
    # Calculate means per prompt family
    # Filter subsets
    unsafe_items = df[df["gold_label"] == "unsafe"]
    safe_items = df[df["gold_label"] == "safe"]
    
    harmful_compliance = unsafe_items.groupby("prompt_family")["compliance_score"].mean().reset_index()
    harmful_compliance.rename(columns={"compliance_score": "harmful_compliance_rate"}, inplace=True)
    
    false_refusals = safe_items.groupby("prompt_family")["refusal_score"].mean().reset_index()
    false_refusals.rename(columns={"refusal_score": "false_refusal_rate"}, inplace=True)
    
    merged = pd.merge(harmful_compliance, false_refusals, on="prompt_family", how="outer").fillna(0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged, 
        x="harmful_compliance_rate", 
        y="false_refusal_rate", 
        hue="prompt_family", 
        s=200, 
        marker='o'
    )
    
    # Add labels
    for i in range(merged.shape[0]):
        plt.text(
            merged["harmful_compliance_rate"][i] + 0.02, 
            merged["false_refusal_rate"][i], 
            merged["prompt_family"][i], 
            horizontalalignment='left'
        )
        
    plt.title("Safety Boundary: Harmful Compliance vs. False Refusal")
    plt.xlabel("Harmful Compliance Rate (lower is better)")
    plt.ylabel("False Refusal Rate (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Safety boundary plot saved to {output_path}")

def plot_hierarchy_conflict(csv_path: str, output_path: str):
    """
    Reads Phase 2 results and plots Accuracy / unsafe continuation rates 
    under vague vs explicit hierarchy prompts evaluating Conflict scenarios.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for conflicts
    conflict_df = df[df["gold_label"] == "conflict"]
    
    agg = conflict_df.groupby("prompt_family")[["hierarchy_following_score", "unsafe_continuation_score"]].mean().reset_index()
    agg.rename(columns={
        "hierarchy_following_score": "Accuracy (Hierarchy Following)",
        "unsafe_continuation_score": "Unsafe Continuation Rate"
    }, inplace=True)
    
    # Melt for grouped bar chart
    melted = agg.melt(id_vars="prompt_family", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="prompt_family", y="Score", hue="Metric")
    
    plt.title("Hierarchy Conflict Performance (IHEval Substrate)")
    plt.xlabel("Prompt Family")
    plt.ylabel("Average Score")
    plt.ylim(0, 1.1)
    plt.legend(title="Metric")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hierarchy conflict plot saved to {output_path}")


# ---------------------------------------------------------------------------
# PRD v2 plots
# ---------------------------------------------------------------------------

def plot_policy_distribution(csv_path: str, output_path: str):
    """Heatmap of policy-label distribution per controller family.

    PRD v2 §5 RQ1: Do prompt-level controllers induce distinct policy
    selections on the same item?
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Cross-tab: rows = prompt_family, cols = classified_policy
    ct = pd.crosstab(df["prompt_family"], df["classified_policy"], normalize="index")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)

    # Annotate cells
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Proportion")
    ax.set_title("Policy Distribution by Controller Family (PRD v2 RQ1)")
    ax.set_xlabel("Classified Policy")
    ax.set_ylabel("Controller Family")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Policy distribution plot saved to {output_path}")


def plot_routing_accuracy(csv_path: str, output_path: str):
    """Grouped bar chart of routing accuracy by family × clarity level.

    PRD v2 §5 RQ2: Does controller clarity improve routing under
    ambiguity and conflict?
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    agg = df.groupby(["prompt_family", "clarity_level"])["routing_correct"].mean().reset_index()
    agg.rename(columns={"routing_correct": "Routing Accuracy"}, inplace=True)

    pivot = agg.pivot(index="prompt_family", columns="clarity_level", values="Routing Accuracy").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Routing Accuracy by Family × Clarity (PRD v2 RQ2)")
    ax.set_xlabel("Controller Family")
    ax.set_ylabel("Routing Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Clarity Level")
    plt.xticks(rotation=45, ha="right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Routing accuracy plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.phase == 1:
        inp = args.input_file or "artifacts/phase1_results.csv"
        out = args.output_file or "artifacts/safety_boundary.png"
        plot_safety_boundary(os.path.join(base, inp), os.path.join(base, out))
    elif args.phase == 2:
        inp = args.input_file or "artifacts/phase2_results.csv"
        out = args.output_file or "artifacts/hierarchy_conflict.png"
        plot_hierarchy_conflict(os.path.join(base, inp), os.path.join(base, out))
