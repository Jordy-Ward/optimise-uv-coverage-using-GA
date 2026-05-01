import numpy as np
import random
import math

class GeneticAlgorithm:
    def __init__(self, num_antennas=64, k=16, pop_size=500, generations=1000, 
                 mutation_rate=0.1, tournament_size=5, niche_radius=4):
        """
        Genetic Algorithm for antenna layout optimization.
        Bumbed parameters for high-resolution Deep Search[cite: 22].
        """
        self.num_antennas = num_antennas
        self.k = k
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.niche_radius = niche_radius

    def initialize_population(self):
        """Creates random 64-bit arrays with exactly k ones[cite: 1, 18]."""
        population = np.zeros((self.pop_size, self.num_antennas), dtype=int)
        for i in range(self.pop_size):
            layout = np.array([1]*self.k + [0]*(self.num_antennas - self.k))
            np.random.shuffle(layout)
            population[i] = layout
        return population

    def apply_fitness_sharing(self, population, raw_fitnesses):
        """Penalizes individuals that are too similar to maintain diversity[cite: 1, 18]."""
        shared_fitnesses = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            distances = np.sum(population[i] != population, axis=1)
            sharing_vals = np.maximum(0, 1 - (distances / self.niche_radius))
            niche_count = np.sum(sharing_vals)
            shared_fitnesses[i] = raw_fitnesses[i] / niche_count
        return shared_fitnesses

    def tournament_selection(self, population, shared_fitnesses):
        """Picks best candidate from a random tournament pool[cite: 1, 18]."""
        combatant_indices = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        best_idx = combatant_indices[np.argmax(shared_fitnesses[combatant_indices])]
        return population[best_idx]

    def crossover(self, parent1, parent2):
        """Intersection crossover guarantees exactly k active antennas[cite: 1, 18]."""
        child = np.zeros(self.num_antennas, dtype=int)
        shared_ones = np.where((parent1 == 1) & (parent2 == 1))[0]
        child[shared_ones] = 1
        
        p1_only = np.where((parent1 == 1) & (parent2 == 0))[0]
        p2_only = np.where((parent1 == 0) & (parent2 == 1))[0]
        disputed_antennas = np.concatenate([p1_only, p2_only])
        
        needed = self.k - len(shared_ones)
        if needed > 0:
            chosen = np.random.choice(disputed_antennas, needed, replace=False)
            child[chosen] = 1
        return child

    def mutate(self, child):
        """Multi-Swap Mutation maintains k=16 constraint[cite: 1, 18]."""
        active_indices = np.where(child == 1)[0].tolist()
        inactive_indices = np.where(child == 0)[0].tolist()
        
        for i in range(len(active_indices)):
            if random.random() < self.mutation_rate:
                turn_off = active_indices[i]
                turn_on = random.choice(inactive_indices)
                
                child[turn_off] = 0
                child[turn_on] = 1
                
                inactive_indices.remove(turn_on)
                inactive_indices.append(turn_off)
        return child
    
    def run(self, fitness_function):
        """Main evolutionary loop with elitism and progress reporting[cite: 1, 18, 22]."""
        population = self.initialize_population()
        best_fitness_history = []
        global_best_layout = None
        global_best_fitness = -float('inf')

        for gen in range(self.generations):
            raw_fitnesses = np.zeros(self.pop_size)
            
            # Skip evaluation for the elite carried over from the previous generation[cite: 18]
            start_idx = 1 if gen > 0 else 0
            if gen > 0:
                raw_fitnesses[0] = global_best_fitness
                
            for i in range(start_idx, self.pop_size):
                raw_fitnesses[i] = fitness_function(population[i])

            current_best_idx = np.argmax(raw_fitnesses)
            if raw_fitnesses[current_best_idx] > global_best_fitness:
                global_best_fitness = raw_fitnesses[current_best_idx]
                global_best_layout = population[current_best_idx].copy()

            best_fitness_history.append(global_best_fitness)
            
            # Progress reporting every 10%[cite: 22]
            if (gen + 1) % (self.generations // 10) == 0:
                print(f"Generation {gen+1}/{self.generations} | Best Fitness: {global_best_fitness:.2f}")

            shared_fitnesses = self.apply_fitness_sharing(population, raw_fitnesses)
            new_population = np.zeros((self.pop_size, self.num_antennas), dtype=int)
            
            # Elitism: Carry over absolute best[cite: 1, 18]
            new_population[0] = global_best_layout

            for i in range(1, self.pop_size):
                parent1 = self.tournament_selection(population, shared_fitnesses)
                parent2 = self.tournament_selection(population, shared_fitnesses)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population[i] = child

            population = new_population

        return global_best_layout, global_best_fitness, best_fitness_history


class SimulatedAnnealing:
    def __init__(self, num_antennas=64, k=16, initial_temp=1000, cooling_rate=0.99995, max_iterations=500000):
        """
        Simulated Annealing for antenna layout optimization.
        Updated cooling rate for high-iteration Deep Search[cite: 22].
        """
        self.num_antennas = num_antennas
        self.k = k
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def get_initial_solution(self):
        """Returns initial layout and tracking lists for faster neighbor generation[cite: 18]."""
        layout = np.zeros(self.num_antennas, dtype=int)
        active_indices = random.sample(range(self.num_antennas), self.k)
        inactive_indices = [i for i in range(self.num_antennas) if i not in active_indices]
        layout[active_indices] = 1
        return layout, active_indices, inactive_indices

    def get_neighbor(self, current_solution, active_indices, inactive_indices):
        """Generates a neighbor by swapping one antenna[cite: 1, 18]."""
        neighbor = current_solution.copy()
        new_active = list(active_indices)
        new_inactive = list(inactive_indices)
        
        turn_off_idx = random.randrange(len(new_active))
        turn_on_idx = random.randrange(len(new_inactive))
        
        turn_off = new_active.pop(turn_off_idx)
        turn_on = new_inactive.pop(turn_on_idx)
        
        neighbor[turn_off] = 0
        neighbor[turn_on] = 1
        
        new_active.append(turn_on)
        new_inactive.append(turn_off)
        return neighbor, new_active, new_inactive
    
    def run(self, fitness_function):
        """Main annealing loop with maximization logic[cite: 1, 18, 22]."""
        current_solution, active_indices, inactive_indices = self.get_initial_solution()
        current_fitness = fitness_function(current_solution)

        best_solution = current_solution.copy()
        best_fitness = current_fitness
        history = []
        temp = self.initial_temp

        for i in range(self.max_iterations):
            neighbor, n_active, n_inactive = self.get_neighbor(current_solution, active_indices, inactive_indices)
            neighbor_fitness = fitness_function(neighbor)
            delta = neighbor_fitness - current_fitness

            # Acceptance probability for maximization[cite: 1, 22]
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                active_indices = n_active
                inactive_indices = n_inactive

            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

            history.append(best_fitness)
            temp *= self.cooling_rate

            # Progress reporting every 10%[cite: 22]
            if (i + 1) % (self.max_iterations // 10) == 0:
                print(f"Iteration {i+1}/{self.max_iterations} | Temp: {temp:.2f} | Best Fitness: {best_fitness:.2f}")

        return best_solution, best_fitness, history