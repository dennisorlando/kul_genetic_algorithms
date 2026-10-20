import sys
import random
import numpy as np
import pandas as pd


class GeneticTSP:
    def __init__(
        self,
        csv_path,
        population_size=100,
        generations=500,
        mutation_rate=0.2,
        elite_size=20,
        penalty=1000,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to adjacency matrix CSV.

        population_size : int
            Number of individuals.

        generations : int
            Number of iterations.

        mutation_rate : float
            Probability of mutating an offspring.

        elite_size : int
            Number of individuals kept during top-k elimination.

        penalty : float
            Penalty applied per invalid edge.
        """

        self.matrix = self.load_matrix(csv_path)

        self.N = len(self.matrix)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.penalty = penalty

    # --------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------

    @staticmethod
    def load_matrix(csv_path):
        """
        Loads adjacency matrix from CSV.

        Invalid connections should be:
            0
            NaN
            negative values

        except diagonal.
        """
        df = pd.read_csv(csv_path, header=None)
        return df.values.astype(float)

    # --------------------------------------------------
    # INDIVIDUAL REPRESENTATION
    # --------------------------------------------------

    def create_individual(self):
        """
        Creates a route:
            [0, perm(...), ...]
        """
        middle = list(range(1, self.N))
        random.shuffle(middle)

        return [0] + middle

    def create_population(self):
        return [
            self.create_individual()
            for _ in range(self.population_size)
        ]

    # --------------------------------------------------
    # FITNESS
    # --------------------------------------------------

    def evaluate(self, individual):
        """
        Computes:

        total_length
        violations
        valid_total_length
        valid_edge_count

        fitness = total_length + penalty * violations
        """

        route = individual + [0]

        total_length = 0.0
        valid_total_length = 0.0
        valid_edge_count = 0
        violations = 0

        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]

            weight = self.matrix[a][b]

            invalid = (
                np.isnan(weight)
                or weight <= 0
            )

            if invalid:
                violations += 1
            else:
                total_length += weight
                valid_total_length += weight
                valid_edge_count += 1

        fitness = total_length + self.penalty * violations

        return {
            "fitness": fitness,
            "total_length": total_length,
            "violations": violations,
            "valid_total_length": valid_total_length,
            "valid_edge_count": valid_edge_count,
        }

    # --------------------------------------------------
    # SELECTION
    # --------------------------------------------------

    def tournament_selection(
        self,
        scored_population,
        tournament_size=3,
    ):
        candidates = random.sample(
            scored_population,
            tournament_size,
        )

        candidates.sort(
            key=lambda x: x["fitness"]
        )

        return candidates[0]["individual"]

    # --------------------------------------------------
    # CROSSOVER
    # --------------------------------------------------

    def crossover(self, parent1, parent2):
        """
        Order crossover (OX).

        Keeps node 0 fixed.
        """

        size = len(parent1)

        start, end = sorted(
            random.sample(range(1, size), 2)
        )

        child = [None] * size
        child[0] = 0

        child[start:end] = parent1[start:end]

        p2_values = [
            x for x in parent2
            if x not in child
        ]

        idx = 1

        for value in p2_values:
            while idx < size and child[idx] is not None:
                idx += 1

            if idx < size:
                child[idx] = value

        return child

    # --------------------------------------------------
    # MUTATION
    # --------------------------------------------------

    def mutate(self, individual):
        """
        Swap mutation.
        """

        if random.random() > self.mutation_rate:
            return individual

        offspring = individual.copy()

        i, j = random.sample(
            range(1, self.N),
            2
        )

        offspring[i], offspring[j] = (
            offspring[j],
            offspring[i]
        )

        return offspring

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    def run(self):
        population = self.create_population()

        best_solution = None
        best_eval = None

        for generation in range(self.generations):

            scored = []

            for individual in population:
                result = self.evaluate(individual)

                result["individual"] = individual
                scored.append(result)

            scored.sort(
                key=lambda x: x["fitness"]
            )

            if (
                best_eval is None
                or scored[0]["fitness"]
                < best_eval["fitness"]
            ):
                best_eval = scored[0].copy()
                best_solution = scored[0]["individual"].copy()

            print(
                f"Gen {generation:4d} | "
                f"Fitness={scored[0]['fitness']:.2f} | "
                f"Length={scored[0]['valid_total_length']:.2f} | "
                f"Violations={scored[0]['violations']}"
            )

            # -------------------------------
            # TOP-K ELIMINATION
            # -------------------------------
            next_population = [
                x["individual"]
                for x in scored[:self.elite_size]
            ]

            while len(next_population) < self.population_size:

                parent1 = self.tournament_selection(scored)
                parent2 = self.tournament_selection(scored)

                child = self.crossover(
                    parent1,
                    parent2,
                )

                child = self.mutate(child)

                next_population.append(child)

            population = next_population

        return best_solution, best_eval


if __name__ == "__main__":

    ga = GeneticTSP(
        csv_path=sys.argv[1],
        population_size=1000,
        generations=5000,
        mutation_rate=0.2,
        elite_size=100,
        penalty=1000,
    )

    best_path, stats = ga.run()

    print("\nBEST ROUTE")
    print(best_path + [0])

    print("\nSTATISTICS")
    print("Fitness:", stats["fitness"])
    print("Valid length:", stats["valid_total_length"])
    print("Violations:", stats["violations"])
    print("Valid edges:", stats["valid_edge_count"])
