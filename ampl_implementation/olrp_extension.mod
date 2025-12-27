# ============================================================
# OLRP model extended: heterogeneous fleet + emissions
# ============================================================

# -----------------------
# Sets
# -----------------------
set DEPOTS;
set CUSTOMERS;
set NODES := DEPOTS union CUSTOMERS;
set VEHICLES;                 # vehicle types v

# -----------------------
# Parameters
# -----------------------

# Vehicle-related parameters
param CV {VEHICLES};          # capacity of vehicle type v
param FV {VEHICLES};          # fixed cost of vehicle type v

# Depot-related parameters
param CD {DEPOTS};            # depot capacities
param FD {DEPOTS};            # depot opening costs

# Customer demand
param d  {CUSTOMERS};         # customer demands

# Coordinates of all nodes
param X {NODES};              # X-coordinate of each node
param Y {NODES};              # Y-coordinate of each node

# Travel cost - distance base (per unit distance cost before scaling for each vehicle type)
# c[i,j] is proportional to distance (Euclidean distance * scale)
param scale default 1;

param c {i in NODES, j in NODES} :=
    if i = j then 0
    else sqrt( (X[i] - X[j])^2 + (Y[i] - Y[j])^2 ) * scale;

# Heterogeneous fleet cost and emissions
param alpha {VEHICLES} default 1;   # economic cost multiplier for vehicle type v: $ kg co2
param EF    {VEHICLES} >= 0;        # emission factor (e.g., kg CO2 per unit distance) for v
param lambda >= 0;                  # weight converting emissions to cost units
param Emax   >= 0;                  # upper bound on total emissions

# Base economic cost per arc and vehicle
param c_base {i in NODES, j in NODES, v in VEHICLES} := alpha[v] * c[i,j];

# Generalized cost including emissions: c~_{ijv} = c_base + lambda * EF[v] * distance
param c_total {i in NODES, j in NODES, v in VEHICLES} := c_base[i,j,v] + lambda * EF[v] * c[i,j];

# -----------------------
# Decision variables
# -----------------------

var x {i in NODES, j in NODES, v in VEHICLES} binary;	# 1 if vehicle type v goes from i to j
var y {k in DEPOTS}                    binary;         	# 1 if depot k is opened
var z {i in CUSTOMERS, k in DEPOTS}   binary;          	# 1 if customer i is assigned to depot k
var U {i in NODES, j in NODES, v in VEHICLES} >= 0;    	# remaining load after leaving i on arc (i,j) with vehicle v

# -----------------------
# Objective function
# -----------------------
# Minimize total cost = generalized travel cost (economic + emissions) + depot opening + fixed vehicle costs

minimize TotalCost:
    # Travel + emission cost: only arcs ending at customers
    sum {i in NODES, j in CUSTOMERS, v in VEHICLES} c_total[i,j,v] * x[i,j,v]
  + # Depot opening cost
    sum {k in DEPOTS} FD[k] * y[k]
  + # Fixed vehicle cost: vehicle of type v used when an arc from depot k to customer i exists
    sum {k in DEPOTS, i in CUSTOMERS, v in VEHICLES} FV[v] * x[k,i,v];


# -----------------------
# Other Constraints
# -----------------------

# (2) Each customer has exactly one outgoing arc
s.t. OneOut {i in CUSTOMERS}:
    sum {j in NODES, v in VEHICLES} x[i,j,v] = 1;

# (3) Flow balance at every node (sum of in-arcs = sum of out-arcs, across all vehicle types;
# 	  also forbidding multiple vehicles in one route)
s.t. FlowBalance {i in NODES, v in VEHICLES}:
    sum {j in NODES} x[j,i,v] = sum {j in NODES} x[i,j,v];

# (4) Demand conservation
s.t. DemandFlow {i in CUSTOMERS}:
    sum {j in NODES, v in VEHICLES} U[j,i,v] - sum {j in NODES, v in VEHICLES} U[i,j,v] = d[i];

# (5) Remaining load cannot exceed vehicle capacity
s.t. VehicleCapacity {i in NODES, j in NODES, v in VEHICLES}:
    U[i,j,v] <= CV[v] * x[i,j,v];

# (6) Total load dispatched from depot equals demand of customers assigned to it
s.t. DepotLoad {k in DEPOTS}:
    sum {j in CUSTOMERS, v in VEHICLES} U[k,j,v] = sum {j in CUSTOMERS} z[j,k] * d[j];

# (7) No remaining load arriving at depot (all demand delivered)
s.t. ZeroLoadAtDepot {k in DEPOTS}:
    sum {j in CUSTOMERS, v in VEHICLES} U[j,k,v] = 0;

# (8) Upper bound on U from a customer (after serving i), per vehicle type
s.t. UpperU {i in CUSTOMERS, j in NODES, v in VEHICLES}:
    U[i,j,v] <= (CV[v] - d[i]) * x[i,j,v];

# (9) Lower bound on U when going to customer j, per vehicle type
s.t. LowerU {i in NODES, j in CUSTOMERS, v in VEHICLES}:
    U[i,j,v] >= d[j] * x[i,j,v];

# (10) Each customer is assigned to exactly one depot
s.t. OneDepotPerCustomer {i in CUSTOMERS}:
    sum {k in DEPOTS} z[i,k] = 1;

# (11) Depot capacity
s.t. DepotCapacity {k in DEPOTS}:
    sum {i in CUSTOMERS} d[i] * z[i,k] <= CD[k] * y[k];

# (12) If an arc goes from customer i to depot k, customer i must be assigned to depot k
s.t. ArcToDepotAllowed {i in CUSTOMERS, k in DEPOTS, v in VEHICLES}:
    x[i,k,v] <= z[i,k];

# (13) If an arc goes from depot k to customer i, customer i must be assigned to depot k
s.t. ArcFromDepotAllowed {i in CUSTOMERS, k in DEPOTS, v in VEHICLES}:
    x[k,i,v] <= z[i,k];

# (14) Prohibit routes that connect customers assigned to different depots
s.t. NoCrossDepotRoutes {i in CUSTOMERS, j in CUSTOMERS, k in DEPOTS: i <> j}:
    sum {v in VEHICLES} x[i,j,v] + z[i,k] + sum {m in DEPOTS diff {k}} z[j,m] <= 2;

# Total emissions computed from distance and emission factor must not exceed Emax -- added
s.t. EmissionLimit:
    sum {i in NODES, j in CUSTOMERS, v in VEHICLES} EF[v] * c[i,j] * x[i,j,v] <= Emax;

# Forbid self-arcs (for any vehicle type) -- added
s.t. NoSelfArc {i in NODES, v in VEHICLES}:
    x[i,i,v] = 0;

# Forbid arcs between depots (for any vehicle type) -- added
s.t. NoDepotToDepot {k in DEPOTS, m in DEPOTS, v in VEHICLES: k <> m}:
    x[k,m,v] = 0;

# U[i,i,v] can be forced to 0 for clarity (for any vehicle type) -- added
s.t. ZeroSelfLoad {i in NODES, v in VEHICLES}:
    U[i,i,v] = 0;
