# Open-Location-Routing-Problem-Extended

This repository contains an **extended Open Location–Routing Problem (OLRP)** that integrates **facility location, customer assignment, and vehicle routing** into a single optimization model.

The project extends the classical OLRP by incorporating **heterogeneous vehicles** and **environmental constraints**, and is implemented in both **AMPL** and **Python (Pyomo)**.

---

## Problem Overview

The model decides:

* Which depots to open
* How to assign customers to depots
* How to route vehicles to serve all customers

with the objective of **minimizing total system cost**.

All decisions are made for a **single static planning horizon**.

---

## Model Extensions

Compared to the standard OLRP, this implementation includes:

* **Heterogeneous vehicle fleet** with different capacities and emission factors
* **Depot capacity constraints**, activated only when a depot is opened
* **Explicit customer–depot assignment**, enforced consistently with routing decisions
* **Environmental component**

  * CO₂ emissions modeled both as a hard constraint and as a cost penalty
* **No cross-depot routing**, ensuring routes are consistent with depot assignments

---

## Implementations

Two equivalent implementations of the same model are provided:

### AMPL

* Mathematical formulation implemented using `.mod`, `.dat`, and `.run` files
* Solved as a Mixed-Integer Programming (MIP) problem with **CPLEX**

### Python / Pyomo

* Model implemented using **Pyomo**
* Allows easier experimentation and post-processing in Python

Both implementations include scripts to **visualize routes**.