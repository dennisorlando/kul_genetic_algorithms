#!/usr/bin/env python3

import argparse
import heapq
import random
from pathlib import Path

import numpy as np
import pandas as pd


class GeneticTSP:
    def __init__(
        self,
        csv_path,
        population_size=1000,
        generations=5000,
        mutation_rate=0.2,
        elite_size=100,
        penalty=1000.0,
        num_swaps=3,
        tournament_size=3,
        report_every=10,
        seed=None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 1 <= elite_size <= population_size:
            raise ValueError("elite_size must be between 1 and population_size")
        if num_swaps < 1:
            raise ValueError("num_swaps must be at least 1")
        if tournament_size < 1 or tournament_size > population_size:
            raise ValueError(
                "tournament_size must be between 1 and population_size"
            )
        if report_every < 1:
            raise ValueError("report_every must be at least 1")

        self.rng = random.Random(seed)
        self.matrix = self.load_matrix(csv_path)
        self.N = self.matrix.shape[0]

        if self.N < 3:
            raise ValueError("The adjacency matrix must contain at least 3 nodes")

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.penalty = float(penalty)
        self.num_swaps = num_swaps
        self.tournament_size = tournament_size
        self.report_every = report_every

    @staticmethod
    def load_matrix(csv_path):
        """Load and validate a square adjacency matrix from a CSV file."""
        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            matrix = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "The CSV must contain only numeric adjacency-matrix values"
            ) from exc

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Adjacency matrix must be square; received shape {matrix.shape}"
            )

        return np.ascontiguousarray(matrix)

    def create_individual(self):
        """Create a permutation starting at node 0; the closing 0 is implicit."""
        middle = list(range(1, self.N))
        self.rng.shuffle(middle)
        return [0] + middle

    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    def evaluate(self, individual):
        """
        Return a tuple containing:
          fitness, valid_total_length, violations, valid_edge_count

        Missing, NaN, zero, and negative edge weights are violations.
        """
        route = np.empty(self.N + 1, dtype=np.intp)
        route[:-1] = individual
        route[-1] = 0

        weights = self.matrix[route[:-1], route[1:]]
        valid_mask = np.isfinite(weights) & (weights > 0.0)

        valid_edge_count = int(np.count_nonzero(valid_mask))
        violations = self.N - valid_edge_count
        valid_total_length = float(weights[valid_mask].sum())
        fitness = valid_total_length + self.penalty * violations

        return (
            fitness,
            valid_total_length,
            violations,
            valid_edge_count,
        )

    def tournament_selection(self, scored_population):
        """Select the best member from a random tournament."""
        candidates = self.rng.sample(
            scored_population,
            self.tournament_size,
        )
        return min(candidates, key=lambda item: item[0])[4]

    def crossover(self, parent1, parent2):
        """Order crossover while keeping node 0 fixed at the first position."""
        size = len(parent1)
        start, end = sorted(self.rng.sample(range(1, size), 2))

        child = [None] * size
        child[0] = 0
        child[start:end] = parent1[start:end]

        used = set(parent1[start:end])
        used.add(0)
        remaining = (node for node in parent2 if node not in used)

        for index in range(1, size):
            if child[index] is None:
                child[index] = next(remaining)

        return child

    def mutate(self, individual):
        """Apply random swap mutations without moving the fixed starting node."""
        if self.rng.random() >= self.mutation_rate:
            return individual

        offspring = individual.copy()

        for _ in range(self.num_swaps):
            i, j = self.rng.sample(range(1, self.N), 2)
            offspring[i], offspring[j] = offspring[j], offspring[i]

        return offspring

    def run(self):
        population = self.create_population()
        best_solution = None
        best_stats = None

        for generation in range(self.generations):
            scored = []

            for individual in population:
                fitness, length, violations, edge_count = self.evaluate(individual)
                scored.append(
                    (fitness, length, violations, edge_count, individual)
                )

            elites = heapq.nsmallest(
                self.elite_size,
                scored,
                key=lambda item: item[0],
            )
            generation_best = elites[0]

            if best_stats is None or generation_best[0] < best_stats[0]:
                best_solution = generation_best[4].copy()
                best_stats = generation_best[:4]

            if (
                generation == 0
                or (generation + 1) % self.report_every == 0
                or generation == self.generations - 1
            ):
                print(
                    f"Gen {generation + 1:5d}/{self.generations} | "
                    f"Fitness={generation_best[0]:.2f} | "
                    f"Length={generation_best[1]:.2f} | "
                    f"Violations={generation_best[2]}"
                )

            # Copy elites so later modifications can never affect stored parents.
            next_population = [elite[4].copy() for elite in elites]

            while len(next_population) < self.population_size:
                parent1 = self.tournament_selection(scored)
                parent2 = self.tournament_selection(scored)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)

            population = next_population

        return best_solution, best_stats


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for a TSP-like adjacency matrix problem."
    )
    parser.add_argument("csv_path", help="Path to the adjacency-matrix CSV file")
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=5000)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elite-size", type=int, default=100)
    parser.add_argument("--penalty", type=float, default=1000.0)
    parser.add_argument("--num-swaps", type=int, default=3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    ga = GeneticTSP(
        csv_path=args.csv_path,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        penalty=args.penalty,
        num_swaps=args.num_swaps,
        tournament_size=args.tournament_size,
        report_every=args.report_every,
        seed=args.seed,
    )

    best_path, stats = ga.run()

    print("\nBEST ROUTE")
    print(best_path + [0])

    print("\nSTATISTICS")
    print("Fitness:", stats[0])
    print("Valid length:", stats[1])
    print("Violations:", stats[2])
    print("Valid edges:", stats[3])


if __name__ == "__main__":
    main()
