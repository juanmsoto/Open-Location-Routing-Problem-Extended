"""Plot routes for the OLRP solution produced by olrp_pyomo.py."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_solution(path: Path):
    data = json.loads(path.read_text())
    # Normalize keys to ints where appropriate
    coords = {int(k): tuple(v) for k, v in data["coords"].items()}
    depots = [int(k) for k in data["depots"]]
    customers = [int(k) for k in data["customers"]]
    arcs = [{"i": int(a["i"]), "j": int(a["j"]), "vehicle": a["vehicle"]} for a in data["arcs"]]
    open_depots = [int(k) for k in data["open_depots"]]
    assignments = [{"customer": int(a["customer"]), "depot": int(a["depot"])} for a in data["assignments"]]

    return {
        "coords": coords,
        "depots": depots,
        "customers": customers,
        "arcs": arcs,
        "open_depots": open_depots,
        "assignments": assignments,
        "vehicles": data.get("vehicles", []),
        "total_cost": data.get("total_cost"),
        "total_emissions": data.get("total_emissions"),
    }


def plot_solution(solution):
    coords = solution["coords"]
    depots = set(solution["depots"])
    customers = set(solution["customers"])
    open_depots = set(solution["open_depots"])
    arcs = solution["arcs"]

    vehicle_colors = {
        "small": "tab:blue",
        "medium": "tab:green",
        "large": "tab:red",
    }

    plt.figure(figsize=(8, 8))

    for k in depots:
        x, y = coords[k]
        face = "yellow" if k in open_depots else "white"
        plt.scatter(x, y, marker="s", s=130, edgecolor="black", facecolor=face, zorder=3)
        plt.text(x + 0.5, y + 0.5, str(k), fontsize=9)

    for i in customers:
        x, y = coords[i]
        plt.scatter(x, y, marker="o", s=45, edgecolor="black", facecolor="white", zorder=3)
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8)

    # Plot only arcs that end at customers (skip arcs ending at depots)
    for arc in arcs:
        i, j, v = arc["i"], arc["j"], arc["vehicle"]
        if j in depots:
            continue
        if i not in coords or j not in coords:
            continue

        x_i, y_i = coords[i]
        x_j, y_j = coords[j]
        color = vehicle_colors.get(v, "gray")

        plt.plot(
            [x_i, x_j],
            [y_i, y_j],
            color=color,
            linewidth=2.4,
            alpha=0.85,
            zorder=2,
        )

    legend_lines = [
        Line2D([0], [0], color=vehicle_colors["small"], lw=2, label="Small vehicle"),
        Line2D([0], [0], color=vehicle_colors["medium"], lw=2, label="Medium vehicle"),
        Line2D([0], [0], color=vehicle_colors["large"], lw=2, label="Large vehicle"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=10,
            label="Open depot",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Customer",
        ),
    ]

    plt.legend(handles=legend_lines, loc="best")
    plt.title("OLRP Solution — Customer Routes Only")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot OLRP solution routes.")
    parser.add_argument("--solution-file", default="solution.json", help="Path to the JSON solution.")
    args = parser.parse_args()

    solution_path = Path(args.solution_file)
    if not solution_path.exists():
        raise SystemExit(f"Solution file not found: {solution_path}")

    solution = load_solution(solution_path)
    plot_solution(solution)


if __name__ == "__main__":
    main()
