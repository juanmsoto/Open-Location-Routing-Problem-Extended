import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# 1. PASTE YOUR AMPL OUTPUT HERE
# ============================================================

ampl_output = """
CPLEX 22.1.1:   lim:time = 600
  mip:gap = 0.029999999999999999
CPLEX 22.1.1: feasible solution; objective 1219.362187
1.92782e+07 simplex iterations
116420 branching nodes
absmipgap=133.869, relmipgap=0.109786

====================
      SOLUTION
====================

Total Cost: 1219.362187
Total Emissions (approx): 303.471588

Open Depots:
  Depot 3
  Depot 5

====================
        ROUTES
====================


Routes starting at depot 3:
Route (vehicle: small): 3 -> 15 -> 13 -> 11 -> 6 -> 7
Route (vehicle: medium): 3 -> 18 -> 16 -> 9 -> 8
Route (vehicle: large): 3 -> 21 -> 19 -> 17 -> 14 -> 12 -> 10

Routes starting at depot 5:
Route (vehicle: large): 5 -> 24 -> 26 -> 22 -> 25 -> 23 -> 20

====================
   DECISION VARIABLES
====================

Depot opening variables y[k]:
  y[1] = 0
  y[2] = 0
  y[3] = 1
  y[4] = 0
  y[5] = 1

Customer-to-depot assignment z[i,k] (only z[i,k] = 1):
  z[6,3] = 1
  z[7,3] = 1
  z[8,3] = 1
  z[9,3] = 1
  z[10,3] = 1
  z[11,3] = 1
  z[12,3] = 1
  z[13,3] = 1
  z[14,3] = 1
  z[15,3] = 1
  z[16,3] = 1
  z[17,3] = 1
  z[18,3] = 1
  z[19,3] = 1
  z[20,5] = 1
  z[21,3] = 1
  z[22,5] = 1
  z[23,5] = 1
  z[24,5] = 1
  z[25,5] = 1
  z[26,5] = 1

Route arc variables x[i,j,v] (only x[i,j,v] = 1):
  x[3,15,small] = 1
  x[3,18,medium] = 1
  x[3,21,large] = 1
  x[5,24,large] = 1
  x[6,7,small] = 1
  x[7,3,small] = 1
  x[8,3,medium] = 1
  x[9,8,medium] = 1
  x[10,3,large] = 1
  x[11,6,small] = 1
  x[12,10,large] = 1
  x[13,11,small] = 1
  x[14,12,large] = 1
  x[15,13,small] = 1
  x[16,9,medium] = 1
  x[17,14,large] = 1
  x[18,16,medium] = 1
  x[19,17,large] = 1
  x[20,5,large] = 1
  x[21,19,large] = 1
  x[22,25,large] = 1
  x[23,20,large] = 1
  x[24,26,large] = 1
  x[25,23,large] = 1
  x[26,22,large] = 1


====================
     END OF RUN
====================`
"""

# ============================================================
# 2. FIXED NODE COORDINATES
# ============================================================

coords = {
    1:  (136, 194), 2:  (143, 237), 3:  (136, 216), 4:  (137, 204), 5:  (128, 197),
    6:  (151, 264), 7:  (159, 261), 8:  (130, 254), 9:  (128, 252), 10: (163, 247),
    11: (146, 246), 12: (161, 242), 13: (142, 239), 14: (163, 236), 15: (148, 232),
    16: (128, 231), 17: (156, 217), 18: (129, 214), 19: (146, 208), 20: (164, 208),
    21: (141, 206), 22: (147, 193), 23: (164, 193), 24: (129, 189), 25: (155, 185),
    26: (139, 182)
}

DEPOTS = {1,2,3,4,5}
CUSTOMERS = set(coords.keys()) - DEPOTS

# ============================================================
# 3. PARSE AMPL OUTPUT
# ============================================================

open_depots = set()
z_assignments = {}
arcs = []

for line in ampl_output.splitlines():
    line = line.strip()

    # y[k] = 1
    m_y = re.match(r"y\[(\d+)\]\s*=\s*(\d+)", line)
    if m_y:
        k = int(m_y.group(1))
        val = int(m_y.group(2))
        if val == 1:
            open_depots.add(k)
        continue

    # z[i,k] = 1
    m_z = re.match(r"z\[(\d+),(\d+)\]\s*=\s*1", line)
    if m_z:
        i = int(m_z.group(1))
        k = int(m_z.group(2))
        z_assignments[i] = k
        continue

    # x[i,j,v] = 1
    m_x = re.match(r"x\[(\d+),(\d+),([A-Za-z_]+)\]\s*=\s*1", line)
    if m_x:
        i = int(m_x.group(1))
        j = int(m_x.group(2))
        v = m_x.group(3)
        arcs.append((i, j, v))
        continue

print("Parsed open depots:", open_depots)
print("Parsed # arcs:", len(arcs))


# ============================================================
# 4. PLOT SOLUTION
# ============================================================

vehicle_colors = {
    "small":  "tab:blue",
    "medium": "tab:green",
    "large":  "tab:red",
}

plt.figure(figsize=(8, 8))

# Depots
for k in DEPOTS:
    x, y = coords[k]
    if k in open_depots:
        plt.scatter(x, y, marker="s", s=130, edgecolor="black", facecolor="yellow", zorder=3)
    else:
        plt.scatter(x, y, marker="s", s=90, edgecolor="black", facecolor="white", zorder=3)
    plt.text(x + 0.5, y + 0.5, str(k), fontsize=9)

# Customers
for i in CUSTOMERS:
    x, y = coords[i]
    plt.scatter(x, y, marker="o", s=45, edgecolor="black", facecolor="white", zorder=3)
    plt.text(x + 0.5, y + 0.5, str(i), fontsize=8)

# Arcs (skip arcs that end at a depot)
for (i, j, v) in arcs:

    if j in DEPOTS:
        continue  # SKIP customer → depot arcs

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
        zorder=2
    )

# Legend
legend_lines = [
    Line2D([0], [0], color=vehicle_colors["small"],  lw=2, label="Small vehicle"),
    Line2D([0], [0], color=vehicle_colors["medium"], lw=2, label="Medium vehicle"),
    Line2D([0], [0], color=vehicle_colors["large"],  lw=2, label="Large vehicle"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="yellow", markeredgecolor="black", markersize=10, label="Open depot"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=8, label="Customer"),
]

plt.legend(handles=legend_lines, loc="best")
plt.title("OLRP Solution — Customer Routes Only")
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
