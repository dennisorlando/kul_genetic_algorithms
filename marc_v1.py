import argparse
import logging
import time

import Reporter
import numpy as np
import numba as nb

log = logging.getLogger("ea")

# Finite stand-in for missing edges (inf). Keeps deltas computable in local search.
# A tour is valid iff its length is < BIG.
BIG = 1e6


# ----------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------

@nb.njit(cache=True)
def tour_length(tour, D):
    n = tour.size
    s = 0.0
    for i in range(n):
        s += D[tour[i], tour[(i + 1) % n]]
    return s


@nb.njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@nb.njit(cache=True)
def nearest_neighbor(D, start):
    n = D.shape[0]
    visited = np.zeros(n, np.bool_)
    tour = np.empty(n, np.int64)
    tour[0] = start
    visited[start] = True
    cur = start
    for k in range(1, n):
        best = -1
        best_d = np.inf
        for c in range(n):
            if not visited[c] and D[cur, c] < best_d:
                best_d = D[cur, c]
                best = c
        tour[k] = best
        visited[best] = True
        cur = best
    return tour


@nb.njit(cache=True)
def order_crossover(p1, p2):
    n = p1.size
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)
    if a > b:
        a, b = b, a
    child = np.empty(n, np.int64)
    used = np.zeros(n, np.bool_)
    for k in range(a, b + 1):
        child[k] = p1[k]
        used[p1[k]] = True
    idx = (b + 1) % n
    for k in range(n):
        c = p2[(b + 1 + k) % n]
        if not used[c]:
            child[idx] = c
            idx = (idx + 1) % n
    return child


@nb.njit(cache=True)
def move_segment(tour, i, L, k):
    """Remove tour[i:i+L] and reinsert it (same orientation) at index k of the remainder."""
    n = tour.size
    out = np.empty(n, np.int64)
    m = 0
    for x in range(n):
        if x < i or x >= i + L:
            if m == k:
                for y in range(L):
                    out[m + y] = tour[i + y]
                m += L
            out[m] = tour[x]
            m += 1
    if k >= n - L:
        for y in range(L):
            out[m + y] = tour[i + y]
    return out


@nb.njit(cache=True)
def insertion_mutation(tour, max_len):
    n = tour.size
    L = np.random.randint(1, max_len + 1)
    i = np.random.randint(0, n - L + 1)
    k = np.random.randint(0, n - L + 1)
    return move_segment(tour, i, L, k)


@nb.njit(cache=True)
def or_opt(tour, D, cand, max_seg, max_passes):
    """First-improvement Or-opt over neighbour lists.

    Segments move without reversal, so the delta is O(1) even for an
    asymmetric D (unlike 2-opt, where reversal changes every inner edge).
    """
    n = tour.size
    pos = np.empty(n, np.int64)
    for p in range(n):
        pos[tour[p]] = p

    for _ in range(max_passes):
        improved = False
        for L in range(1, max_seg + 1):
            i = 0
            while i <= n - L:
                p_city = tour[(i - 1) % n]
                nx_city = tour[(i + L) % n]
                s0 = tour[i]
                s1 = tour[i + L - 1]
                remove_gain = D[p_city, s0] + D[s1, nx_city] - D[p_city, nx_city]
                if remove_gain > 1e-9:
                    for t in range(cand.shape[1]):
                        c = cand[s0, t]
                        j = pos[c]
                        if (j - i) % n < L or j == (i - 1) % n:
                            continue
                        d = tour[(j + 1) % n]
                        add_cost = D[c, s0] + D[s1, d] - D[c, d]
                        if add_cost < remove_gain - 1e-9:
                            k = j if j < i else j - L
                            tour = move_segment(tour, i, L, k + 1)
                            for p in range(n):
                                pos[tour[p]] = p
                            improved = True
                            break
                i += 1
        if not improved:
            break
    return tour


@nb.njit(cache=True)
def tournament(fit, k):
    best = np.random.randint(0, fit.size)
    for _ in range(k - 1):
        c = np.random.randint(0, fit.size)
        if fit[c] < fit[best]:
            best = c
    return best


@nb.njit(cache=True)
def make_offspring(pop, fit, D, cand, lam, k, mut_rate, ls_prob):
    n = pop.shape[1]
    off = np.empty((lam, n), np.int64)
    off_fit = np.empty(lam)
    for o in range(lam):
        a = tournament(fit, k)
        b = tournament(fit, k)
        child = order_crossover(pop[a], pop[b])
        if np.random.random() < mut_rate:
            child = insertion_mutation(child, 5)
        if np.random.random() < ls_prob:
            child = or_opt(child, D, cand, 3, 3)
        off[o] = child
        off_fit[o] = tour_length(child, D)
    return off, off_fit


@nb.njit(cache=True)
def init_population(D, cand, mu, nn_fraction):
    n = D.shape[0]
    pop = np.empty((mu, n), np.int64)
    fit = np.empty(mu)
    n_nn = int(mu * nn_fraction)
    for i in range(mu):
        if i < n_nn:
            t = nearest_neighbor(D, np.random.randint(0, n))
        else:
            t = np.random.permutation(n)
        t = or_opt(t, D, cand, 3, 5)
        pop[i] = t
        fit[i] = tour_length(t, D)
    return pop, fit


@nb.njit(cache=True)
def perturb(pop, fit, D, cand, keep, strength):
    """Restart diversity: keep the best `keep` individuals (pop sorted), re-randomise the rest."""
    for i in range(keep, pop.shape[0]):
        t = pop[i].copy()
        for _ in range(strength):
            t = insertion_mutation(t, 10)
        t = or_opt(t, D, cand, 3, 3)
        pop[i] = t
        fit[i] = tour_length(t, D)


# ----------------------------------------------------------------------
# Elimination (survivor selection)
# ----------------------------------------------------------------------

def eliminate(pop, fit, off, off_fit, mu):
    """(mu + lambda) truncation, skipping clones (same fitness) to preserve diversity."""
    all_pop = np.vstack((pop, off))
    all_fit = np.concatenate((fit, off_fit))
    order = np.argsort(all_fit, kind="stable")

    keep = []
    last = -np.inf
    for idx in order:
        if all_fit[idx] - last > 1e-9:
            keep.append(idx)
            last = all_fit[idx]
            if len(keep) == mu:
                break
    if len(keep) < mu:  # not enough distinct fitness values: top up with the best remaining
        chosen = set(keep)
        for idx in order:
            if idx not in chosen:
                keep.append(idx)
                if len(keep) == mu:
                    break
    keep = np.array(keep)
    keep = keep[np.argsort(all_fit[keep], kind="stable")]
    return all_pop[keep], all_fit[keep]


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------

# Modify the class name to match your student number.
class TSPSolver:

    def __init__(self, mu=None, seed=42, patience=None, log_every=200):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.mu = mu                # population size (None = chosen from n)
        self.seed = seed
        self.patience = patience    # stop after this many restarts without a new best (0 = never, None = max(20, n//2))
        self.log_every = log_every  # log a progress line every N generations

    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        D = np.where(np.isfinite(distanceMatrix), distanceMatrix, BIG).astype(np.float64)
        n = D.shape[0]

        # Candidate lists: K cities c with the cheapest edge c -> s (insertion "after c").
        K = min(10, n - 1)
        Dc = D.copy()
        np.fill_diagonal(Dc, np.inf)
        cand = np.argsort(Dc, axis=0)[:K].T.copy().astype(np.int64)

        mu = self.mu or (100 if n <= 500 else 80)
        lam = mu
        tournament_k = 3
        mut_rate = 0.2
        ls_prob = 0.3
        stagnation_limit = 50
        patience = max(20, n // 2) if self.patience is None else self.patience

        t0 = time.time()
        log.info("loaded %s: n=%d mu=%d lambda=%d K=%d", filename, n, mu, lam, K)
        seed_rng(self.seed)
        pop, fit = init_population(D, cand, mu, 0.5)  # first call also pays the JIT compile
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        log.info("init done in %.1fs, best=%.3f", time.time() - t0, fit[0])

        best_fit = fit[0]
        stagnation = 0
        restarts_without_best = 0
        generation = 0

        while True:
            off, off_fit = make_offspring(
                pop, fit, D, cand, lam, tournament_k, mut_rate, ls_prob
            )
            pop, fit = eliminate(pop, fit, off, off_fit, mu)

            generation += 1
            if fit[0] < best_fit - 1e-9:
                best_fit = fit[0]
                stagnation = 0
                restarts_without_best = 0
                log.debug("gen %d new best %.3f", generation, best_fit)
            else:
                stagnation += 1

            if stagnation >= stagnation_limit:
                restarts_without_best += 1
                log.debug("gen %d restart #%d without new best", generation, restarts_without_best)
                perturb(pop, fit, D, cand, max(2, mu // 10), 5)
                order = np.argsort(fit)
                pop, fit = pop[order], fit[order]
                stagnation = 0

            if generation % self.log_every == 0:
                log.info("gen %6d | %6.1fs | best %.3f | mean %.3f",
                         generation, time.time() - t0, fit[0], fit.mean())

            meanObjective = float(fit.mean())
            bestObjective = float(fit[0])
            bestSolution = pop[0]

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # Leave the next three lines as they are.
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                log.info("time budget used up")
                break
            # Convergence test: many restarts in a row without a new global best.
            if patience and restarts_without_best >= patience:
                log.info("converged: %d restarts without improvement", restarts_without_best)
                break

        log.info("finished: %d generations, %.1fs, best %.3f", generation, time.time() - t0, best_fit)
        return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Evolutionary algorithm for the asymmetric TSP")
    ap.add_argument("file", nargs="?", default="data/tour50.csv", help="distance matrix CSV")
    ap.add_argument("-t", "--time", type=float, default=300, help="time budget in seconds")
    ap.add_argument("-m", "--mu", type=int, default=None, help="population size (default: from n)")
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-p", "--patience", type=int, default=None,
                    help="stop after this many restarts without a new best (0 = use the full time, default max(20, n//2))")
    ap.add_argument("--log-every", type=int, default=200, help="progress line every N generations")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    a = TSPSolver(mu=args.mu, seed=args.seed, patience=args.patience, log_every=args.log_every)
    a.reporter.allowedTime = args.time
    a.optimize(args.file)

