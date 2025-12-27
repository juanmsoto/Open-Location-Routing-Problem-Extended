"""
Pyomo implementation of the OLRP model (heterogeneous fleet + emissions).
Builds the model, solves it, prints a summary, saves a JSON solution, and
optionally plots the routes.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pyomo.environ as pyo

DATA = {
    "DEPOTS": [1, 2, 3, 4, 5],
    "CUSTOMERS": list(range(6, 27)),
    "VEHICLES": ["small", "medium", "large"],
    "coords": {
        1: (136, 194),
        2: (143, 237),
        3: (136, 216),
        4: (137, 204),
        5: (128, 197),
        6: (151, 264),
        7: (159, 261),
        8: (130, 254),
        9: (128, 252),
        10: (163, 247),
        11: (146, 246),
        12: (161, 242),
        13: (142, 239),
        14: (163, 236),
        15: (148, 232),
        16: (128, 231),
        17: (156, 217),
        18: (129, 214),
        19: (146, 208),
        20: (164, 208),
        21: (141, 206),
        22: (147, 193),
        23: (164, 193),
        24: (129, 189),
        25: (155, 185),
        26: (139, 182),
    },
    "d": {
        6: 1100,
        7: 700,
        8: 800,
        9: 1400,
        10: 2100,
        11: 400,
        12: 800,
        13: 100,
        14: 500,
        15: 600,
        16: 1200,
        17: 1300,
        18: 1300,
        19: 300,
        20: 900,
        21: 2100,
        22: 1000,
        23: 900,
        24: 2500,
        25: 1800,
        26: 700,
    },
    "CD": {k: 15000 for k in range(1, 6)},
    "FD": {k: 50 for k in range(1, 6)},
    "CV": {"small": 3000, "medium": 5000, "large": 8000},
    "FV": {"small": 120, "medium": 160, "large": 220},
    "alpha": {"small": 0.70, "medium": 1.00, "large": 1.15},
    "EF": {"small": 0.40, "medium": 0.75, "large": 1.80},
    "lambda": 0.5,
    "Emax": 400,
    "scale": 1.0,
}


def build_costs(data):
    coords = data["coords"]
    scale = data["scale"]
    nodes = data["DEPOTS"] + data["CUSTOMERS"]

    c = {}
    c_base = {}
    c_total = {}

    for i in nodes:
        xi, yi = coords[i]
        for j in nodes:
            if i == j:
                dist = 0.0
            else:
                xj, yj = coords[j]
                dist = math.hypot(xi - xj, yi - yj) * scale
            c[(i, j)] = dist
            for v in data["VEHICLES"]:
                base = data["alpha"][v] * dist
                total = base + data["lambda"] * data["EF"][v] * dist
                c_base[(i, j, v)] = base
                c_total[(i, j, v)] = total

    return c, c_base, c_total


# -----------------------
# Model construction
# -----------------------

def build_model(data):
    c, c_base, c_total = build_costs(data)

    m = pyo.ConcreteModel(name="OLRP_Heterogeneous_Emissions")

    # Sets
    m.DEPOTS = pyo.Set(initialize=data["DEPOTS"])
    m.CUSTOMERS = pyo.Set(initialize=data["CUSTOMERS"])
    m.NODES = pyo.Set(initialize=data["DEPOTS"] + data["CUSTOMERS"])
    m.VEHICLES = pyo.Set(initialize=data["VEHICLES"])

    # Parameters
    m.CV = pyo.Param(m.VEHICLES, initialize=data["CV"])
    m.FV = pyo.Param(m.VEHICLES, initialize=data["FV"])
    m.CD = pyo.Param(m.DEPOTS, initialize=data["CD"])
    m.FD = pyo.Param(m.DEPOTS, initialize=data["FD"])
    m.d = pyo.Param(m.CUSTOMERS, initialize=data["d"])
    m.X = pyo.Param(m.NODES, initialize={k: data["coords"][k][0] for k in data["coords"]})
    m.Y = pyo.Param(m.NODES, initialize={k: data["coords"][k][1] for k in data["coords"]})
    m.scale = pyo.Param(initialize=data["scale"])
    m.alpha = pyo.Param(m.VEHICLES, initialize=data["alpha"])
    m.EF = pyo.Param(m.VEHICLES, initialize=data["EF"])
    m.lmbda = pyo.Param(initialize=data["lambda"])
    m.Emax = pyo.Param(initialize=data["Emax"])

    m.c = pyo.Param(m.NODES, m.NODES, initialize=c, within=pyo.NonNegativeReals)
    m.c_base = pyo.Param(m.NODES, m.NODES, m.VEHICLES, initialize=c_base, within=pyo.NonNegativeReals)
    m.c_total = pyo.Param(m.NODES, m.NODES, m.VEHICLES, initialize=c_total, within=pyo.NonNegativeReals)

    # Decision variables
    m.x = pyo.Var(m.NODES, m.NODES, m.VEHICLES, domain=pyo.Binary)
    m.y = pyo.Var(m.DEPOTS, domain=pyo.Binary)
    m.z = pyo.Var(m.CUSTOMERS, m.DEPOTS, domain=pyo.Binary)
    m.U = pyo.Var(m.NODES, m.NODES, m.VEHICLES, domain=pyo.NonNegativeReals)

    # Objective
    def total_cost_rule(model):
        travel = sum(
            model.c_total[i, j, v] * model.x[i, j, v]
            for i in model.NODES
            for j in model.CUSTOMERS
            for v in model.VEHICLES
        )
        open_cost = sum(model.FD[k] * model.y[k] for k in model.DEPOTS)
        fixed_vehicle = sum(
            model.FV[v] * model.x[k, i, v]
            for k in model.DEPOTS
            for i in model.CUSTOMERS
            for v in model.VEHICLES
        )
        return travel + open_cost + fixed_vehicle

    m.TotalCost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # Constraints
    def one_out_rule(model, i):
        return sum(model.x[i, j, v] for j in model.NODES for v in model.VEHICLES) == 1

    m.OneOut = pyo.Constraint(m.CUSTOMERS, rule=one_out_rule)

    def flow_balance_rule(model, i, v):
        return sum(model.x[j, i, v] for j in model.NODES) == sum(model.x[i, j, v] for j in model.NODES)

    m.FlowBalance = pyo.Constraint(m.NODES, m.VEHICLES, rule=flow_balance_rule)

    def demand_flow_rule(model, i):
        inflow = sum(model.U[j, i, v] for j in model.NODES for v in model.VEHICLES)
        outflow = sum(model.U[i, j, v] for j in model.NODES for v in model.VEHICLES)
        return inflow - outflow == model.d[i]

    m.DemandFlow = pyo.Constraint(m.CUSTOMERS, rule=demand_flow_rule)

    def vehicle_capacity_rule(model, i, j, v):
        return model.U[i, j, v] <= model.CV[v] * model.x[i, j, v]

    m.VehicleCapacity = pyo.Constraint(m.NODES, m.NODES, m.VEHICLES, rule=vehicle_capacity_rule)

    def depot_load_rule(model, k):
        return sum(model.U[k, j, v] for j in model.CUSTOMERS for v in model.VEHICLES) == sum(
            model.z[j, k] * model.d[j] for j in model.CUSTOMERS
        )

    m.DepotLoad = pyo.Constraint(m.DEPOTS, rule=depot_load_rule)

    def zero_load_at_depot_rule(model, k):
        return sum(model.U[j, k, v] for j in model.CUSTOMERS for v in model.VEHICLES) == 0

    m.ZeroLoadAtDepot = pyo.Constraint(m.DEPOTS, rule=zero_load_at_depot_rule)

    def upper_u_rule(model, i, j, v):
        return model.U[i, j, v] <= (model.CV[v] - model.d[i]) * model.x[i, j, v]

    m.UpperU = pyo.Constraint(m.CUSTOMERS, m.NODES, m.VEHICLES, rule=upper_u_rule)

    def lower_u_rule(model, i, j, v):
        return model.U[i, j, v] >= model.d[j] * model.x[i, j, v]

    m.LowerU = pyo.Constraint(m.NODES, m.CUSTOMERS, m.VEHICLES, rule=lower_u_rule)

    def one_depot_per_customer_rule(model, i):
        return sum(model.z[i, k] for k in model.DEPOTS) == 1

    m.OneDepotPerCustomer = pyo.Constraint(m.CUSTOMERS, rule=one_depot_per_customer_rule)

    def depot_capacity_rule(model, k):
        return sum(model.d[i] * model.z[i, k] for i in model.CUSTOMERS) <= model.CD[k] * model.y[k]

    m.DepotCapacity = pyo.Constraint(m.DEPOTS, rule=depot_capacity_rule)

    def arc_to_depot_rule(model, i, k, v):
        return model.x[i, k, v] <= model.z[i, k]

    m.ArcToDepotAllowed = pyo.Constraint(m.CUSTOMERS, m.DEPOTS, m.VEHICLES, rule=arc_to_depot_rule)

    def arc_from_depot_rule(model, i, k, v):
        return model.x[k, i, v] <= model.z[i, k]

    m.ArcFromDepotAllowed = pyo.Constraint(m.CUSTOMERS, m.DEPOTS, m.VEHICLES, rule=arc_from_depot_rule)

    def no_cross_depot_routes_rule(model, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        return (
            sum(model.x[i, j, v] for v in model.VEHICLES)
            + model.z[i, k]
            + sum(model.z[j, m_] for m_ in model.DEPOTS if m_ != k)
            <= 2
        )

    m.NoCrossDepotRoutes = pyo.Constraint(m.CUSTOMERS, m.CUSTOMERS, m.DEPOTS, rule=no_cross_depot_routes_rule)

    def emission_limit_rule(model):
        return (
            sum(model.EF[v] * model.c[i, j] * model.x[i, j, v] for i in model.NODES for j in model.CUSTOMERS for v in model.VEHICLES)
            <= model.Emax
        )

    m.EmissionLimit = pyo.Constraint(rule=emission_limit_rule)

    def no_self_arc_rule(model, i, v):
        return model.x[i, i, v] == 0

    m.NoSelfArc = pyo.Constraint(m.NODES, m.VEHICLES, rule=no_self_arc_rule)

    def no_depot_to_depot_rule(model, k, m_, v):
        if k == m_:
            return pyo.Constraint.Skip
        return model.x[k, m_, v] == 0

    m.NoDepotToDepot = pyo.Constraint(m.DEPOTS, m.DEPOTS, m.VEHICLES, rule=no_depot_to_depot_rule)

    def zero_self_load_rule(model, i, v):
        return model.U[i, i, v] == 0

    m.ZeroSelfLoad = pyo.Constraint(m.NODES, m.VEHICLES, rule=zero_self_load_rule)

    return m


# -----------------------
# Solving and extraction
# -----------------------

def solve_model(model, solver_name="highs", time_limit=600, mip_gap=0.03):
    solver_name = solver_name.lower()
    user_requested = solver_name != "highs"

    if solver_name == "highs":
        try:
            from pyomo.contrib.appsi.solvers import Highs

            opt = Highs()
            if hasattr(opt, "options"):
                opt.options["time_limit"] = time_limit
                opt.options["mip_rel_gap"] = mip_gap
            else:
                opt.config.time_limit = time_limit
                opt.config.mip_gap = mip_gap
            results = opt.solve(model, load_solutions=True)
            return results, "appsi_highs"
        except Exception as exc:  
            print(f"[info] Appsi HiGHS not available ({exc}); trying command-line highs if present.", file=sys.stderr)

        opt = pyo.SolverFactory("highs")
        if opt is not None and opt.available(False):
            options = {"time_limit": time_limit, "mip_rel_gap": mip_gap}
            results = opt.solve(model, tee=False, options=options)
            return results, "highs"

        solver_name = "cbc"

    candidates = [solver_name]
    if solver_name != "cbc":
        candidates.append("cbc")
    if solver_name != "glpk":
        candidates.append("glpk")

    for cand in candidates:
        opt = pyo.SolverFactory(cand)
        if opt is None or not opt.available(False):
            continue

        options = {}
        if cand == "cbc":
            options.update({"seconds": time_limit, "ratioGap": mip_gap})
        elif cand == "glpk":
            options.update({"tmlim": time_limit, "mipgap": mip_gap})
        elif cand == "cplex":
            options.update({"timelimit": time_limit, "mipgap": mip_gap})
        elif cand == "gurobi":
            options.update({"TimeLimit": time_limit, "MIPGap": mip_gap})
        else:
            options.update({"timelimit": time_limit, "mipgap": mip_gap})

        results = opt.solve(model, tee=False, options=options)
        return results, cand

    if user_requested:
        raise RuntimeError(f"Requested solver '{solver_name}' is not available.")

    raise RuntimeError(
        "No solver found. Install one of: highspy (HiGHS), coin-or-cbc, glpk, or use CPLEX/Gurobi if licensed.\n"
        "Examples:\n"
        "  pip install highspy\n"
        "  brew install cbc\n"
        "  brew install glpk\n"
        "  # CPLEX: install IBM Python API and ensure 'cplex' is on PATH\n"
        "  # Gurobi: install gurobipy and ensure license is set"
    )


def compute_emissions(model):
    return sum(
        pyo.value(model.EF[v]) * pyo.value(model.c[i, j]) * pyo.value(model.x[i, j, v])
        for i in model.NODES
        for j in model.CUSTOMERS
        for v in model.VEHICLES
    )


def extract_solution(model, data):
    total_cost = pyo.value(model.TotalCost)
    total_emissions = compute_emissions(model)

    open_depots = [int(k) for k in model.DEPOTS if pyo.value(model.y[k]) > 0.5]
    assignments = [
        {"customer": int(i), "depot": int(k)}
        for i in model.CUSTOMERS
        for k in model.DEPOTS
        if pyo.value(model.z[i, k]) > 0.5
    ]
    arcs = [
        {"i": int(i), "j": int(j), "vehicle": str(v)}
        for i in model.NODES
        for j in model.NODES
        for v in model.VEHICLES
        if pyo.value(model.x[i, j, v]) > 0.5
    ]

    solution = {
        "total_cost": float(total_cost),
        "total_emissions": float(total_emissions),
        "open_depots": open_depots,
        "assignments": assignments,
        "arcs": arcs,
        "coords": data["coords"],
        "depots": data["DEPOTS"],
        "customers": data["CUSTOMERS"],
        "vehicles": data["VEHICLES"],
    }
    return solution


def print_summary(solution):
    print("\n====================")
    print("      SOLUTION")
    print("====================\n")
    print(f"Total Cost: {solution['total_cost']:.6f}")
    print(f"Total Emissions (approx): {solution['total_emissions']:.6f}\n")

    print("Open Depots:")
    for k in solution["open_depots"]:
        print(f"  Depot {k}")
    print()

    print("====================")
    print("   DECISION VARIABLES")
    print("====================\n")

    print("Depot opening variables y[k]:")
    for k in solution["depots"]:
        flag = 1 if k in solution["open_depots"] else 0
        print(f"  y[{k}] = {flag}")
    print()

    print("Customer-to-depot assignment z[i,k] (only z[i,k] = 1):")
    for assign in solution["assignments"]:
        print(f"  z[{assign['customer']},{assign['depot']}] = 1")
    print()

    print("Route arc variables x[i,j,v] (only x[i,j,v] = 1):")
    for arc in solution["arcs"]:
        print(f"  x[{arc['i']},{arc['j']},{arc['vehicle']}] = 1")
    print()


# -----------------------
# Main / CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Solve the OLRP in Pyomo and plot routes.")
    parser.add_argument("--solver", default="highs", help="Solver to use: highs (preferred), cbc, glpk, cplex, gurobi.")
    parser.add_argument("--time-limit", type=float, default=600.0, help="Time limit in seconds.")
    parser.add_argument("--mip-gap", type=float, default=0.03, help="Relative MIP gap.")
    parser.add_argument("--solution-file", default="solution.json", help="Path to write the JSON solution.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting even if matplotlib is available.")
    args = parser.parse_args()

    model = build_model(DATA)

    try:
        results, used_solver = solve_model(model, solver_name=args.solver, time_limit=args.time_limit, mip_gap=args.mip_gap)
    except Exception as exc:  
        print(f"[error] Solver failed: {exc}")
        sys.exit(1)

    termination = getattr(results.solver, "termination_condition", "unknown")
    print(f"[info] Solver used: {used_solver}; termination: {termination}")

    solution = extract_solution(model, DATA)
    print_summary(solution)

    solution_path = Path(args.solution_file)
    solution_path.write_text(json.dumps(solution, indent=2))
    print(f"\nSaved solution to {solution_path.resolve()}")

    if not args.no_plot:
        try:
            import plot_routes

            plot_routes.plot_solution(solution)
        except Exception as exc:  
            print(f"[warning] Could not plot solution automatically: {exc}")


if __name__ == "__main__":
    main()
