

import time
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

print("Loading matrix...", flush=True)
cost = np.loadtxt("tour500.csv", delimiter=",")
n = cost.shape[0]

assert cost.shape == (n, n), f"Expected square matrix, got {cost.shape}"
assert np.isfinite(cost).all(), "Matrix contains NaN or infinite values"

scale = 1000
cost_int = np.rint(cost * scale).astype(np.int64)

print(f"Loaded {n}x{n} matrix.", flush=True)
print("Building routing model...", flush=True)

manager = pywrapcp.RoutingIndexManager(n, 1, 0)
routing = pywrapcp.RoutingModel(manager)

def arc_cost(from_index, to_index):
    i = manager.IndexToNode(from_index)
    j = manager.IndexToNode(to_index)
    return int(cost_int[i, j])

callback_index = routing.RegisterTransitCallback(arc_cost)
routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
params.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
params.time_limit.seconds = 60

# Enable OR-Tools' own search logging
params.log_search = True

# Track each newly accepted solution
start = time.perf_counter()
best_cost = float("inf")
solution_count = 0

def on_solution():
    global best_cost, solution_count
    solution_count += 1
    current = routing.CostVar().Value() / scale

    if current < best_cost:
        best_cost = current

    elapsed = time.perf_counter() - start
    print(
        f"[{elapsed:6.1f}s] Solution #{solution_count}: "
        f"cost={current:.3f}, best={best_cost:.3f}",
        flush=True,
    )

routing.AddAtSolutionCallback(on_solution)

print("Starting search (60-second limit)...", flush=True)
solution = routing.SolveWithParameters(params)
elapsed = time.perf_counter() - start

if solution is None:
    print(f"No solution found after {elapsed:.1f}s.", flush=True)
    raise SystemExit(1)

route = []
idx = routing.Start(0)

while not routing.IsEnd(idx):
    route.append(manager.IndexToNode(idx))
    idx = solution.Value(routing.NextVar(idx))

route.append(manager.IndexToNode(idx))

total_cost = sum(cost[a, b] for a, b in zip(route[:-1], route[1:]))

print("\nSearch finished!", flush=True)
print(f"Nodes:          {n}")
print(f"Solutions seen: {solution_count}")
print(f"Tour cost:      {total_cost:.6f}")
print(f"Search time:    {elapsed:.2f}s")

with open("ortools_tour.txt", "w") as f:
    f.write(str(route))

print("Saved tour to ortools_tour.csv", flush=True)
