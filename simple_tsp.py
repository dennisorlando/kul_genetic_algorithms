import time
import numpy as np
import pandas as pd
import argparse

class GeneticTSP:
    """Same GA as your version (OX crossover, swap mutation, tournament selection,
    elitism), but the population is an (P, n) numpy array and every cost is computed
    with numpy instead of Python loops."""

    def __init__(
        self,
        distance_matrix,
        population_size=200,
        generations=1000,
        crossover_rate=0.9,
        mutation_rate=0.1,
        elite_size=10,
        tournament_k=5,
        nn_seed_frac=0.0,       # fraction of the initial population built with nearest-neighbour
        inf_penalty=1e12,
        seed=None,
    ):
        D = np.asarray(distance_matrix, dtype=float)
        self.n = len(D)
        # replace Inf once here, instead of calling np.isinf on every edge of every route
        self.Dp = np.where(np.isfinite(D), D, inf_penalty)
        self.inf_penalty = inf_penalty

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.k = tournament_k
        self.nn_seed_frac = nn_seed_frac
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------
    # FITNESS: whole population in one numpy call
    # ----------------------------------------------------
    def costs(self, pop):
        nxt = np.roll(pop, -1, axis=1)           # city after each city (wraps around)
        return self.Dp[pop, nxt].sum(axis=1)

    # ----------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------
    def nearest_neighbour_tour(self, start):
        seen = np.zeros(self.n, dtype=bool)
        tour = np.empty(self.n, dtype=int)
        cur = start
        for k in range(self.n):
            tour[k] = cur
            seen[cur] = True
            if k == self.n - 1:
                break
            cur = int(np.where(seen, np.inf, self.Dp[cur]).argmin())
        return tour

    def initialize_population(self):
        pop = np.array([self.rng.permutation(self.n)
                        for _ in range(self.population_size)])
        m = int(self.nn_seed_frac * self.population_size)
        if m > 0:
            starts = self.rng.choice(self.n, m, replace=False)
            for i, s in enumerate(starts):
                pop[i] = self.nearest_neighbour_tour(s)
        return pop

    # ----------------------------------------------------
    # SELECTION: all tournaments at once
    # ----------------------------------------------------
    def tournament_indices(self, costs, m):
        cand = self.rng.integers(0, len(costs), size=(m, self.k))
        return cand[np.arange(m), costs[cand].argmin(axis=1)]

    # ----------------------------------------------------
    # ORDER CROSSOVER (OX), O(n) with a boolean mask
    # (your `city not in child` was O(n) per city -> O(n^2) per crossover)
    # ----------------------------------------------------
    def crossover(self, p1, p2):
        n = self.n
        a, b = np.sort(self.rng.choice(n, 2, replace=False))
        used = np.zeros(n, dtype=bool)
        used[p1[a:b]] = True
        p2_rot = np.roll(p2, -b)                  # parent-2 order, starting after the 2nd cut
        rest = p2_rot[~used[p2_rot]]
        child = np.empty(n, dtype=p1.dtype)
        child[a:b] = p1[a:b]
        child[b:] = rest[: n - b]
        child[:a] = rest[n - b:]
        return child

    # ----------------------------------------------------
    # MUTATION (SWAP)
    # ----------------------------------------------------
    def mutate(self, route):
        i, j = self.rng.choice(self.n, 2, replace=False)
        route[i], route[j] = route[j], route[i]

    # ----------------------------------------------------
    # EVOLUTION
    # ----------------------------------------------------
    def next_generation(self, pop, costs):
        P = self.population_size
        order = np.argsort(costs)
        new_pop = np.empty_like(pop)
        new_pop[: self.elite_size] = pop[order[: self.elite_size]]

        m = P - self.elite_size
        i1 = self.tournament_indices(costs, m)
        i2 = self.tournament_indices(costs, m)
        do_x = self.rng.random(m) < self.crossover_rate
        do_m = self.rng.random(m) < self.mutation_rate

        for c in range(m):
            if do_x[c]:
                child = self.crossover(pop[i1[c]], pop[i2[c]])
            else:
                child = pop[i1[c]].copy()
            if do_m[c]:
                self.mutate(child)
            new_pop[self.elite_size + c] = child
        return new_pop

    # ----------------------------------------------------
    # RUN
    # ----------------------------------------------------
    def run(self, verbose_every=50, time_limit=None):
        pop = self.initialize_population()
        costs = self.costs(pop)
        best_cost = costs.min()
        best_route = pop[costs.argmin()].copy()
        t0 = time.time()

        for g in range(self.generations):
            pop = self.next_generation(pop, costs)
            costs = self.costs(pop)
            j = costs.argmin()
            if costs[j] < best_cost:
                best_cost = costs[j]
                best_route = pop[j].copy()

            if verbose_every and g % verbose_every == 0:
                n_inf = int(best_cost // self.inf_penalty)   # >0 means the route uses Inf edges
                print(f"Gen {g:5d} | best {best_cost:13.2f} | mean {costs.mean():13.2f} "
                      f"| inf edges in best: {n_inf} | {time.time() - t0:6.1f}s", flush=True)
            if time_limit and time.time() - t0 > time_limit:
                break

        return best_route, best_cost


def load_instance(csv_file):
    return pd.read_csv(csv_file, header=None).values.astype(float)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/tour500.csv")
    args = parser.parse_args()
    
    matrix = load_instance(args.file)
    ga = GeneticTSP(
        matrix,
        population_size=250,
        generations=1000,
        crossover_rate=0.9,
        mutation_rate=0.15,
        elite_size=15,
        nn_seed_frac=0.2,      # set to 0.0 to get your original random-only initialisation
        seed=0,
    )
    best_route, best_cost = ga.run()

    assert sorted(best_route) == list(range(len(matrix))), "not a valid permutation"
    print("\nBest cost:", best_cost)
    print("Inf edges used:", int(best_cost // ga.inf_penalty))
