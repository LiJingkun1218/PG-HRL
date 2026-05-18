import numpy as np
import random
from typing import List, Tuple, Dict, Set, Optional, Any
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from functools import lru_cache
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from common.cfunctions import (decode_schedule,evaluate_objectives)
@dataclass
class Solution:
    """Solution data structure."""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Dict = None
    skill_factor: Optional[int] = None  # Skill factor (MFEA)
    transfer_quality: float = 0.0  # Transfer quality

class JobShopProblem:
    """Job shop scheduling problem definition."""
    
    def __init__(self, num_jobs=50, num_machines=5, arrival_time=None, due_date=None, 
                 machine_sequence=None, processing_times=None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.arrival_time = arrival_time if arrival_time is not None else [None]*num_jobs
        self.due_date = due_date if due_date is not None else [None]*num_jobs
        self.machine_sequence = machine_sequence if machine_sequence is not None else [[] for _ in range(num_jobs)]
        self.processing_times = processing_times if processing_times is not None else [[] for _ in range(num_jobs)]
        self.jobs_data = self.generate_random_instance()
        
    def generate_random_instance(self):
        """Generate a random instance."""
        jobs_data = []
        
        for job_id in range(self.num_jobs):
            # Job arrival time
            arrival_time = random.randint(0, 20) if self.arrival_time[job_id] is None else self.arrival_time[job_id]
            
            # Job due date
            due_date = arrival_time + random.randint(50, 150) if self.due_date[job_id] is None else self.due_date[job_id]
            
            # Machine sequence per job
            if len(self.machine_sequence[job_id]) == 0:
                machine_sequence = random.sample(range(self.num_machines), self.num_machines)
            else:
                machine_sequence = self.machine_sequence[job_id]
            
            # Processing times
            if len(self.processing_times[job_id]) == 0:
                processing_times = [random.randint(1, 20) for _ in range(self.num_machines)]
            else:
                processing_times = self.processing_times[job_id]
            
            jobs_data.append({
                'job_id': job_id,
                'arrival_time': arrival_time,
                'due_date': due_date,
                'machine_sequence': machine_sequence,
                'processing_times': processing_times,
                'total_processing_time': sum(processing_times)
            })
        
        return jobs_data

@dataclass
class DynamicEnvironment:
    """Dynamic environment description."""
    time: int
    event_type: str
    parameters: Dict[str, Any]
    similarity: float = 0.0  # Similarity to history

@dataclass
class KnowledgeBase:
    """Knowledge base entry."""
    environment_features: np.ndarray
    optimal_solutions: List[Solution]
    performance_metrics: Dict[str, float]
    transfer_matrix: Optional[np.ndarray] = None  # Transfer relation matrix

class Tr_DMOEAScheduler:
    """Tr-DMOEA implementation."""
    
    def __init__(self, problem: JobShopProblem, dynamic_events: List[Dict] = None,
                 pop_size=100, max_gen=50, archive_size=50,
                 transfer_pool_size=20, prediction_horizon=3, batch_size=10, cache_size=500):
        """
        Initialize Tr-DMOEA scheduler
        Args:
            problem: JobShopProblem instance
            dynamic_events: dynamic event list
            pop_size: population size
            max_gen_per_batch: max generations per batch
            archive_size: archive size
            transfer_pool_size: transfer pool size
            prediction_horizon: prediction horizon
            batch_size: batch size
            cache_size: cache size
        """
        # Problem instance
        self.problem = problem
        
        # Dynamic events
        self.dynamic_events = dynamic_events if dynamic_events else []
        
        # Algorithm parameters
        self.pop_size = pop_size
        self.max_gen_per_batch = max_gen
        self.archive_size = archive_size
        self.transfer_pool_size = transfer_pool_size
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        
        # Tr-DMOEA core components
        self.population = []  # Current population
        self.global_archive = []  # Global archive
        self.knowledge_base = []  # Knowledge base
        self.transfer_pool = []  # Transfer pool
        self.environment_history = []  # Environment history
        self.current_environment = None  # Current environment
        
        # Transfer learning parameters
        self.transfer_probability = 0.7  # Transfer probability
        self.transfer_adaptation_rate = 0.1  # Transfer adaptation rate
        
        # Environment predictor
        self.environment_predictor = None
        self._init_environment_predictor()
        
        # Cache config
        self.cache_size = cache_size
        self._decode_cache = lru_cache(maxsize=cache_size)(self._decode_schedule_core)
        
        # Algorithm state
        self.current_time = 0
        self.generation_count = 0
        self.dynamic_change_count = 0

    def _init_environment_predictor(self):
        """Initialize environment predictor (Gaussian process)."""
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.environment_predictor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-4
        )

    def _decode_schedule_core(self, individual_tuple: Tuple[int], num_jobs: int, 
                             num_machines: int, jobs_tuple: Tuple) -> Dict:
        """Key: decode core logic."""
        individual = list(individual_tuple)
        
        # Restore jobs_data from tuple
        jobs_data = []
        for job_info in jobs_tuple:
            job_id, arrival_time, due_date, machine_sequence, processing_times, total_processing_time = job_info
            jobs_data.append({
                'job_id': job_id,
                'arrival_time': arrival_time,
                'due_date': due_date,
                'machine_sequence': list(machine_sequence),
                'processing_times': list(processing_times),
                'total_processing_time': total_processing_time
            })
        
        # Validate chromosome
        for job_id in individual:
            if job_id is None or not isinstance(job_id, int) or job_id < 0 or job_id >= num_jobs:
                return self._create_invalid_schedule(num_machines, num_jobs)
        
        # Initialize schedules and job progress
        machine_schedules = [[] for _ in range(num_machines)]
        job_progress = [{
            'current_op': 0,
            'completion_time': jobs_data[i]['arrival_time']
        } for i in range(num_jobs)]
        
        # Decode logic
        individual_copy = individual[:]
        scheduled_count = 0
        total_operations = len(individual_copy)
        
        while scheduled_count < total_operations:
            for pos in range(len(individual_copy)):
                job_id = individual_copy[pos]
                if job_id == -1:
                    continue
                    
                job_data = jobs_data[job_id]
                op_index = job_progress[job_id]['current_op']
                
                if op_index >= num_machines:
                    continue
                    
                machine_id = job_data['machine_sequence'][op_index]
                processing_time = job_data['processing_times'][op_index]
                job_ready_time = job_progress[job_id]['completion_time']
                machine_ready_time = machine_schedules[machine_id][-1]['end_time'] if machine_schedules[machine_id] else 0
                
                start_time = max(job_ready_time, machine_ready_time)
                end_time = start_time + processing_time
                
                # Record operation info
                operation_info = {
                    'job_id': job_id,
                    'operation_index': op_index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'processing_time': processing_time
                }
                
                machine_schedules[machine_id].append(operation_info)
                job_progress[job_id]['completion_time'] = end_time
                job_progress[job_id]['current_op'] += 1
                individual_copy[pos] = -1
                scheduled_count += 1
        
        return {
            'machine_schedules': machine_schedules,
            'job_completion_times': [job_progress[i]['completion_time'] for i in range(num_jobs)],
            'valid': True
        }

    def _create_invalid_schedule(self, num_machines: int, num_jobs: int) -> Dict:
        """Create invalid schedule."""
        max_value = float('inf')
        return {
            'machine_schedules': [[] for _ in range(num_machines)],
            'job_completion_times': [max_value for _ in range(num_jobs)],
            'valid': False
        }

    def extract_environment_features(self, problem_instance: JobShopProblem) -> np.ndarray:
        """Extract environment feature vector."""
        jobs_data = problem_instance.jobs_data
        
        if not jobs_data:
            return np.zeros(5)
        
        # Extract key features
        processing_times = [d['total_processing_time'] for d in jobs_data]
        due_dates = [d['due_date'] for d in jobs_data]
        arrival_times = [d['arrival_time'] for d in jobs_data]
        
        features = np.array([
            np.mean(processing_times),  # Mean processing time
            np.std(processing_times),   # Std of processing time
            np.mean(due_dates) - np.mean(arrival_times),  # Mean slack
            len(jobs_data) / problem_instance.num_machines,  # Machine load
            self._count_dynamic_jobs(jobs_data) / len(jobs_data)  # Dynamic job ratio
        ])
        
        return features

    def _count_dynamic_jobs(self, jobs_data: List[Dict]) -> int:
        """Count dynamic jobs."""
        return sum(1 for d in jobs_data if d.get('dynamic_flag', False))

    def create_individual(self, problem_instance: JobShopProblem) -> List[int]:
        """Create individual (operation-based encoding)."""
        individual = []
        for job_id in range(problem_instance.num_jobs):
            individual.extend([job_id] * problem_instance.num_machines)
        random.shuffle(individual)
        return individual

    def initialize_population(self, problem_instance: JobShopProblem) -> List[Solution]:
        """Initialize population."""
        population = []
        
        # 1. Rule-based initialization
        for _ in range(self.pop_size // 3):
            individual = self._initialize_by_rule(problem_instance, 'SPT')
            schedule = decode_schedule(self,individual)
            objectives = evaluate_objectives(self,schedule)
            population.append(Solution(
                individual=individual,
                objectives=objectives,
                schedule=schedule
            ))
        
        # 2. Random initialization
        for _ in range(self.pop_size // 3):
            individual = self.create_individual(problem_instance)
            schedule = decode_schedule(self,individual)
            objectives = evaluate_objectives(self,schedule)
            population.append(Solution(
                individual=individual,
                objectives=objectives,
                schedule=schedule
            ))
        
        # 3. Transfer-based initialization
        for _ in range(self.pop_size // 3 + self.pop_size % 3):
            individual = self._initialize_by_transfer(problem_instance)
            schedule = decode_schedule(self,individual)
            objectives = evaluate_objectives(self,schedule)
            population.append(Solution(
                individual=individual,
                objectives=objectives,
                schedule=schedule,
                transfer_quality=0.5
            ))
        
        return population

    def _initialize_by_rule(self, problem_instance: JobShopProblem, rule: str) -> List[int]:
        """Initialize by rule."""
        jobs_data = problem_instance.jobs_data
        
        if rule == 'SPT':
            # Short processing time first
            jobs_sorted = sorted(range(problem_instance.num_jobs), 
                               key=lambda x: jobs_data[x]['total_processing_time'])
        elif rule == 'EDD':
            # Earliest due date first
            jobs_sorted = sorted(range(problem_instance.num_jobs),
                               key=lambda x: jobs_data[x]['due_date'])
        else:
            # Random order
            jobs_sorted = list(range(problem_instance.num_jobs))
            random.shuffle(jobs_sorted)
        
        individual = []
        for job_id in jobs_sorted:
            individual.extend([job_id] * problem_instance.num_machines)
        
        random.shuffle(individual)  # Keep some randomness
        return individual

    def _initialize_by_transfer(self, problem_instance: JobShopProblem) -> List[int]:
        """Initialize with transfer knowledge (ensure feasibility)."""
        if not self.knowledge_base:
            return self.create_individual(problem_instance)
        
        # Find most similar environment
        current_features = self.extract_environment_features(problem_instance)
        similarities = []
        
        for kb in self.knowledge_base:
            similarity = self._calculate_similarity(current_features, kb.environment_features)
            similarities.append((similarity, kb))
        
        similarities.sort(reverse=True)
        
        # Select from best knowledge base
        _, best_kb = similarities[0]
        if best_kb.optimal_solutions:
            # Select best historical solution
            best_solution = min(best_kb.optimal_solutions, 
                            key=lambda x: sum(x.objectives))
            
            # Adapt and ensure feasibility
            adapted_individual = self._adapt_transferred_solution(
                best_solution.individual, 
                problem_instance.num_jobs
            )
            # Repair for feasibility
            adapted_individual = self._repair_individual(adapted_individual, problem_instance)
            return adapted_individual
        
        return self.create_individual(problem_instance)

    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute environment similarity."""
        if np.linalg.norm(features1) == 0 or np.linalg.norm(features2) == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
        )
        
        # Feature weights
        weights = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        weighted_similarity = similarity * np.exp(-np.sum(weights * np.abs(features1 - features2)))
        
        return float(weighted_similarity)

    def _adapt_transferred_solution(self, source_individual: List[int], 
                               target_job_count: int) -> List[int]:
        """Adapt transferred solution (ensure feasibility)."""
        num_machines = self.problem.num_machines
        target_length = target_job_count * num_machines
        
        if len(source_individual) == target_length:
            # Same length
            return source_individual.copy()
        elif len(source_individual) > target_length:
            # Truncate with feasibility fix
            truncated = source_individual[:target_length]
            # Count after truncation
            job_counts = [0] * target_job_count
            for gene in truncated:
                if 0 <= gene < target_job_count:
                    job_counts[gene] += 1
            
            # Already feasible
            if all(count == num_machines for count in job_counts[:target_job_count]):
                return truncated
            
            # Otherwise regenerate a feasible one
            return self._create_feasible_individual(target_job_count)
        else:
            # Extend to ensure each job appears num_machines times
            result = []
            for job_id in range(target_job_count):
                result.extend([job_id] * num_machines)
            random.shuffle(result)
            return result
        
    def _create_feasible_individual(self, num_jobs: int) -> List[int]:
        """Create a feasible individual."""
        individual = []
        for job_id in range(num_jobs):
            individual.extend([job_id] * self.problem.num_machines)
        random.shuffle(individual)
        return individual

    def dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if solution 1 dominates solution 2."""
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better

    def fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sort."""
        if not solutions:
            return [[]]
            
        n = len(solutions)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        # Build solution index map
        sol_to_idx = {id(sol): idx for idx, sol in enumerate(solutions)}
        
        for i in range(n):
            sol_i = solutions[i]
            for j in range(n):
                if i == j:
                    continue
                sol_j = solutions[j]
                
                if self.dominates(sol_i.objectives, sol_j.objectives):
                    dominated_solutions[i].append(j)
                elif self.dominates(sol_j.objectives, sol_i.objectives):
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(sol_i)
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for sol in fronts[current_front]:
                idx = sol_to_idx[id(sol)]
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(solutions[dominated_idx])
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts

    def _calculate_crowding_distance(self, front: List[Solution]) -> List[float]:
        """Crowding distance."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        n = len(front)
        distances = [0.0] * n
        
        for m in range(3):  # Three objectives
            sorted_indices = sorted(range(n), key=lambda i: front[i].objectives[m])
            
            # Boundary solutions
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            min_obj = front[sorted_indices[0]].objectives[m]
            max_obj = front[sorted_indices[-1]].objectives[m]
            obj_range = max_obj - min_obj if max_obj > min_obj else 1.0
            
            for i in range(1, n - 1):
                prev_obj = front[sorted_indices[i-1]].objectives[m]
                next_obj = front[sorted_indices[i+1]].objectives[m]
                distances[sorted_indices[i]] += (next_obj - prev_obj) / obj_range
        
        return distances

    def transfer_learning_operation(self, population: List[Solution], 
                                   problem_instance: JobShopProblem) -> List[Solution]:
        """Transfer learning operation."""
        if not self.knowledge_base or random.random() > self.transfer_probability:
            return []
        
        # Select source knowledge
        current_features = self.extract_environment_features(problem_instance)
        source_kb = self._select_source_knowledge(current_features)
        
        if not source_kb or not source_kb.optimal_solutions:
            return []
        
        # Transfer pool size control
        transfer_count = min(len(source_kb.optimal_solutions), 
                           self.transfer_pool_size // 2)
        
        # Select individuals to transfer
        transferred_solutions = []
        for sol in random.sample(source_kb.optimal_solutions, transfer_count):
            # Adaptation
            adapted_individual = self._adapt_transferred_solution(
                sol.individual, 
                problem_instance.num_jobs
            )
            
            # Evaluate
            schedule = decode_schedule(self,adapted_individual)
            objectives = evaluate_objectives(self,schedule)
            
            transferred_solutions.append(Solution(
                individual=adapted_individual,
                objectives=objectives,
                schedule=schedule,
                transfer_quality=self._evaluate_transfer_quality(sol, objectives)
            ))
        
        return transferred_solutions

    def _select_source_knowledge(self, current_features: np.ndarray) -> Optional[KnowledgeBase]:
        """Select source knowledge base."""
        if not self.knowledge_base:
            return None
        
        # Guard: validate current_features
        if current_features is None or np.all(current_features == 0):
            return None
        
        # Compute similarity
        similarities = []
        for kb in self.knowledge_base:
            # Validate kb features
            if kb.environment_features is None or np.all(kb.environment_features == 0):
                continue
                
            similarity = self._calculate_similarity(current_features, kb.environment_features)
            similarities.append((similarity, kb))
        
        # Select highest similarity
        if similarities:
            similarities.sort(reverse=True)
            return similarities[0][1]
        
        return None

    def _evaluate_transfer_quality(self, source_solution: Solution, 
                                  target_objectives: Tuple[float, ...]) -> float:
        """Evaluate transfer quality."""
        # Simple quality metric: objective improvement
        source_sum = sum(source_solution.objectives)
        target_sum = sum(target_objectives)
        
        if source_sum > 0:
            improvement = (source_sum - target_sum) / source_sum
            return max(0, min(1, 0.5 + improvement * 0.5))
        
        return 0.5

    def sbx_crossover(self, parent1: List[int], parent2: List[int], problem_instance: JobShopProblem) -> Tuple[List[int], List[int]]:
        """SBX crossover (keeps feasibility)."""
        n = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Standard SBX
        eta_c = 20
        crossover_prob = 0.9
        
        for i in range(n):
            if random.random() < crossover_prob:
                u = random.random()
                
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                
                # Apply crossover
                child1[i] = int(0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]))
                child2[i] = int(0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i]))
                
                # Clamp to valid range
                max_job = problem_instance.num_jobs - 1
                child1[i] = max(0, min(child1[i], max_job))
                child2[i] = max(0, min(child2[i], max_job))
        
        # Repair to ensure each job appears num_machines times
        child1 = self._repair_individual(child1, problem_instance)
        child2 = self._repair_individual(child2, problem_instance)
        
        return child1, child2

    def polynomial_mutation(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Polynomial mutation (keeps feasibility)."""
        n = len(individual)
        mutant = individual.copy()
        
        eta_m = 20
        mutation_prob = 1.0 / n
        
        for i in range(n):
            if random.random() < mutation_prob:
                u = random.random()
                
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                
                # Apply mutation
                new_value = int(individual[i] + delta * individual[i])
                new_value = max(0, min(new_value, problem_instance.num_jobs - 1))
                mutant[i] = new_value
        
        # Repair after mutation
        mutant = self._repair_individual(mutant, problem_instance)
        return mutant

    def _repair_individual(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Repair individual to ensure job counts."""
        num_jobs = problem_instance.num_jobs
        num_machines = problem_instance.num_machines
        
        # Count job occurrences
        job_counts = [0] * num_jobs
        for gene in individual:
            if 0 <= gene < num_jobs:
                job_counts[gene] += 1
        
        # Repair strategy
        repaired = individual.copy()
        
        # Strategy 1: fill missing jobs
        for job_id in range(num_jobs):
            required = num_machines
            current = job_counts[job_id]
            
            if current < required:
                # Need to increase count
                to_add = required - current
                # Replace over-represented jobs
                for i in range(len(repaired)):
                    if to_add <= 0:
                        break
                    gene = repaired[i]
                    if job_counts[gene] > num_machines:
                        repaired[i] = job_id
                        job_counts[gene] -= 1
                        job_counts[job_id] += 1
                        to_add -= 1
        
        # Strategy 2: fix remaining issues
        job_counts_final = [0] * num_jobs
        for gene in repaired:
            if 0 <= gene < num_jobs:
                job_counts_final[gene] += 1
        
        for job_id in range(num_jobs):
            required = num_machines
            current = job_counts_final[job_id]
            
            if current != required:
                # Direct adjustment
                diff = current - required
                if diff > 0:
                    # Need to reduce
                    indices = [i for i, g in enumerate(repaired) if g == job_id]
                    for i in indices[:diff]:
                        # Replace with missing job
                        for target_job in range(num_jobs):
                            if job_counts_final[target_job] < num_machines:
                                repaired[i] = target_job
                                job_counts_final[job_id] -= 1
                                job_counts_final[target_job] += 1
                                break
                elif diff < 0:
                    # Need to increase
                    indices = [i for i, g in enumerate(repaired) if job_counts_final[g] > num_machines]
                    for i in indices[:abs(diff)]:
                        old_job = repaired[i]
                        repaired[i] = job_id
                        job_counts_final[old_job] -= 1
                        job_counts_final[job_id] += 1
        
        return repaired

    def generate_offspring(self, population: List[Solution], 
                      problem_instance: JobShopProblem) -> List[Solution]:
        """Generate offspring with repaired genetic operators."""
        offspring = []
        
        # Tournament selection
        tournament_size = 3
        
        while len(offspring) < len(population):
            # Select parents
            parents = random.sample(population, tournament_size)
            parents.sort(key=lambda x: sum(x.objectives))
            parent1, parent2 = parents[0], parents[1]
            
            # Crossover (with repair)
            child1_genes, child2_genes = self.sbx_crossover(
                parent1.individual, 
                parent2.individual,
                problem_instance  # Added parameter
            )
            
            # Mutation (with repair)
            child1_genes = self.polynomial_mutation(child1_genes, problem_instance)  # Added parameter
            child2_genes = self.polynomial_mutation(child2_genes, problem_instance)  # Added parameter
            
            # Evaluate offspring
            for child_genes in [child1_genes, child2_genes]:
                if len(offspring) >= len(population):
                    break
                
                # Final feasibility check
                job_counts = [0] * problem_instance.num_jobs
                for gene in child_genes:
                    if 0 <= gene < problem_instance.num_jobs:
                        job_counts[gene] += 1
                
                if all(count == problem_instance.num_machines for count in job_counts):
                    schedule = decode_schedule(self,child_genes)
                    objectives = evaluate_objectives(self,schedule)
                    
                    offspring.append(Solution(
                        individual=child_genes,
                        objectives=objectives,
                        schedule=schedule
                    ))
        
        return offspring

    def environmental_selection(self, population: List[Solution], 
                               offspring: List[Solution]) -> List[Solution]:
        """Environmental selection."""
        combined = population + offspring
        
        # Non-dominated sort
        fronts = self.fast_non_dominated_sort(combined)
        
        # Select survivors
        new_population = []
        remaining = self.pop_size
        
        for front in fronts:
            if len(front) <= remaining:
                new_population.extend(front)
                remaining -= len(front)
            else:
                # Select by crowding distance
                distances = self._calculate_crowding_distance(front)
                sorted_indices = np.argsort(distances)[::-1]  # Descending
                
                for i in range(remaining):
                    new_population.append(front[sorted_indices[i]])
                break
        
        return new_population

    def update_knowledge_base(self, population: List[Solution], 
                             problem_instance: JobShopProblem):
        """Update knowledge base."""
        current_features = self.extract_environment_features(problem_instance)
        
        # Extract non-dominated solutions
        fronts = self.fast_non_dominated_sort(population)
        if fronts:
            non_dominated_solutions = fronts[0]
            
            # Performance metrics
            objectives = np.array([sol.objectives for sol in non_dominated_solutions])
            metrics = {
                'avg_objectives': np.mean(objectives, axis=0).tolist(),
                'best_objectives': np.min(objectives, axis=0).tolist(),
                'diversity': self._calculate_diversity(non_dominated_solutions),
                'convergence': self._calculate_convergence(non_dominated_solutions)
            }
            
            # Create knowledge entry
            knowledge = KnowledgeBase(
                environment_features=current_features.copy(),
                optimal_solutions=non_dominated_solutions[:10],  # Keep top 10
                performance_metrics=metrics
            )
            
            # Append to knowledge base
            self.knowledge_base.append(knowledge)
            
            # Enforce knowledge base size
            if len(self.knowledge_base) > 20:
                self.knowledge_base.pop(0)

    def _calculate_diversity(self, solutions: List[Solution]) -> float:
        """Compute population diversity."""
        if len(solutions) <= 1:
            return 0.0
        
        objectives = np.array([sol.objectives for sol in solutions])
        
        # Mean nearest-neighbor distance
        if len(objectives) > 1:
            distances = []
            for i in range(len(objectives)):
                min_dist = float('inf')
                for j in range(len(objectives)):
                    if i != j:
                        dist = np.linalg.norm(objectives[i] - objectives[j])
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            return float(np.mean(distances))
        
        return 0.0

    def _calculate_convergence(self, solutions: List[Solution]) -> float:
        """Compute population convergence."""
        if not solutions:
            return float('inf')
        
        # Mean distance to ideal point
        ideal_point = np.min([sol.objectives for sol in solutions], axis=0)
        distances = [np.linalg.norm(np.array(sol.objectives) - ideal_point) 
                    for sol in solutions]
        
        return float(np.mean(distances))

    def predict_environment_change(self, problem_instance: JobShopProblem) -> Optional[np.ndarray]:
        """Predict environment change."""
        # Check prediction prerequisites
        if len(self.environment_history) < self.prediction_horizon:
            return None
        
        # Prepare training data
        X = []
        y = []
        
        for i in range(len(self.environment_history) - 1):
            hist_i = self.environment_history[i]
            hist_j = self.environment_history[i + 1]
            
            # Validate history
            if hist_i is None or hist_j is None:
                continue
                
            X.append(hist_i)
            y.append(hist_j)
        
        if len(X) == 0 or len(y) == 0:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Train predictor
        try:
            self.environment_predictor.fit(X, y)
            
            # Current environment features
            current_env = self.extract_environment_features(problem_instance)
            if current_env is None:
                return None
                
            # Predict next environment
            next_env = self.environment_predictor.predict([current_env])[0]
            return next_env
        except Exception as e:
            print(f"Environment prediction failed: {e}")
            return None

    def adaptive_parameter_adjustment(self):
        """Adaptive parameter adjustment."""
        # Adjust transfer probability by change frequency
        change_frequency = self.dynamic_change_count / (self.current_time + 1)
        
        # More changes -> higher transfer probability
        self.transfer_probability = min(0.9, 0.5 + 0.4 * change_frequency)
        
        # Adjust transfer adaptation rate
        if len(self.knowledge_base) > 5:
            # Adjust by recent success rate
            recent_knowledge = self.knowledge_base[-5:]
            success_rates = []
            
            for kb in recent_knowledge:
                if hasattr(kb, 'transfer_success_rate'):
                    success_rates.append(kb.transfer_success_rate)
            
            if success_rates:
                avg_success_rate = np.mean(success_rates)
                self.transfer_adaptation_rate = 0.05 + 0.15 * avg_success_rate

    def detect_dynamic_change(self, problem_instance: JobShopProblem) -> bool:
        """Detect dynamic change."""
        # Guard: current_environment
        if self.current_environment is None:
            self.current_environment = self.extract_environment_features(problem_instance)
            return True
        
        new_environment = self.extract_environment_features(problem_instance)
        
        # First call check
        if self.current_environment is None and new_environment is None:
            return False
        
        # Change magnitude
        change_magnitude = np.linalg.norm(new_environment - self.current_environment)
        
        # Significant change threshold
        change_threshold = 0.2
        if change_magnitude > change_threshold:
            self.current_environment = new_environment.copy()  # Avoid reference issues
            self.environment_history.append(new_environment.copy())
            self.dynamic_change_count += 1
            return True
        
        return False

    def _evolve_batch(self, problem_instance: JobShopProblem) -> Dict:
        """Run evolution process."""
        # Initialize population
        if not self.population:
            self.population = self.initialize_population(problem_instance)
        
        # Main evolution loop
        for gen in range(self.max_gen_per_batch):
            # Transfer learning
            transferred_solutions = self.transfer_learning_operation(
                self.population, problem_instance
            )
            
            # Generate offspring
            offspring = self.generate_offspring(self.population, problem_instance)
            
            # Environmental selection
            self.population = self.environmental_selection(
                self.population, offspring + transferred_solutions
            )
            
            # Update global archive
            self.global_archive = self.update_archive(
                self.global_archive, 
                self.population
            )
            
            self.generation_count += 1
            
            # Progress log
            if gen % 10 == 0 and self.max_gen_per_batch > 20:
                best_front = self.fast_non_dominated_sort(self.population)[0]
                avg_objectives = np.mean([sol.objectives for sol in best_front], axis=0)
                print(f"Generation {gen}: Best front size={len(best_front)}, "
                      f"Avg objectives={avg_objectives}")
        
        # Select batch-best
        best_sol = None
        if self.population:
            # Normalize objectives to avoid scale bias
            objs = np.array([sol.objectives for sol in self.population])
            norm_objs = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-6)
            best_idx = np.argmin(norm_objs.sum(axis=1))
            best_sol = self.population[best_idx]
        
        return {
            'archive': self.global_archive,
            'final_population': self.population,
            'best_solution': best_sol
        }

    def update_archive(self, archive: List[Solution], 
                      new_solutions: List[Solution]) -> List[Solution]:
        """Update global archive."""
        combined = archive + new_solutions
        fronts = self.fast_non_dominated_sort(combined)
        
        new_archive = []
        remaining = self.archive_size
        
        for front in fronts:
            if len(front) <= remaining:
                new_archive.extend(front)
                remaining -= len(front)
            else:
                # Select by crowding distance
                distances = self._calculate_crowding_distance(front)
                sorted_indices = np.argsort(distances)[::-1]
                
                for i in range(remaining):
                    new_archive.append(front[sorted_indices[i]])
                break
        
        return new_archive

    def _merge_batch_schedule(self, global_schedule: Dict, batch_schedule: Dict, 
                             batch_job_indices: List[int], time_offset: int):
        """Merge batch schedule into global schedule."""
        # Merge machine schedules
        for machine_id in range(len(global_schedule['machine_schedules'])):
            batch_machine_ops = batch_schedule['machine_schedules'][machine_id]
            # Apply time offset
            for op in batch_machine_ops:
                global_op = op.copy()
                # Map batch job_id to global job_id
                global_op['job_id'] = batch_job_indices[op['job_id']]
                global_op['start_time'] += time_offset
                global_op['end_time'] += time_offset
                global_schedule['machine_schedules'][machine_id].append(global_op)
        
        # Merge job completion times
        for batch_job_id, global_job_id in enumerate(batch_job_indices):
            completion_time = batch_schedule['job_completion_times'][batch_job_id]
            global_schedule['job_completion_times'][global_job_id] = completion_time + time_offset
        
        # Update global makespan
        global_schedule['makespan'] = max(global_schedule['job_completion_times'])

    def run(self) -> Dict:
        """Run Tr-DMOEA and return non-dominated set."""
        start_time = time.time()
        
        print("Starting Tr-DMOEA batch optimization...")
        print(f"Total jobs: {self.problem.num_jobs}, Batch size: {self.batch_size}")
        
        # Initialize global schedule
        all_jobs = list(range(self.problem.num_jobs))
        num_batches = (len(all_jobs) + self.batch_size - 1) // self.batch_size
        global_schedule = {
            'machine_schedules': [[] for _ in range(self.problem.num_machines)],
            'job_completion_times': [0] * self.problem.num_jobs,
            'makespan': 0
        }
        time_offset = 0
        batch_results = []
        
        for batch_idx in range(num_batches):
            # Batch job indices
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(all_jobs))
            batch_job_indices = all_jobs[batch_start:batch_end]
            batch_size = len(batch_job_indices)
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: jobs {batch_start}-{batch_end-1} ({batch_size} total)")
            
            # Create batch problem instance
            batch_arrival = []
            batch_due = []
            batch_machine_seq = []
            batch_process_times = []
            
            for local_idx, global_job_id in enumerate(batch_job_indices):
                job_data = self.problem.jobs_data[global_job_id]
                batch_arrival.append(job_data['arrival_time'] + time_offset)
                batch_due.append(job_data['due_date'] + time_offset)
                batch_machine_seq.append(job_data['machine_sequence'])
                batch_process_times.append(job_data['processing_times'])
            
            batch_problem = JobShopProblem(
                num_jobs=batch_size,
                num_machines=self.problem.num_machines,
                arrival_time=batch_arrival,
                due_date=batch_due,
                machine_sequence=batch_machine_seq,
                processing_times=batch_process_times
            )
            
            # Detect dynamic change
            dynamic_change = self.detect_dynamic_change(batch_problem)
            if dynamic_change:
                print("  Dynamic change detected, updating transfer parameters")
                
                # Environment prediction
                predicted_env = self.predict_environment_change(batch_problem)
                if predicted_env is not None:
                    print(f"  Predicted environment features: {predicted_env}")
                
                # Adaptive parameter adjustment
                self.adaptive_parameter_adjustment()
            
            # Batch optimization
            batch_start_time = time.time()
            batch_result = self._evolve_batch(batch_problem)
            batch_elapsed = time.time() - batch_start_time
            
            # Merge batch schedule to global
            if batch_result['best_solution']:
                best_schedule = batch_result['best_solution'].schedule
                self._merge_batch_schedule(global_schedule, best_schedule, batch_job_indices, time_offset)
                
                # Update time offset
                time_offset = global_schedule['makespan']
            
            # Update knowledge base
            self.update_knowledge_base(batch_result['final_population'], batch_problem)
            
            batch_results.append({
                'batch_idx': batch_idx,
                'archive_size': len(batch_result['archive']),
                'elapsed_time': batch_elapsed,
                'best_objectives': batch_result['best_solution'].objectives if batch_result['best_solution'] else None
            })
            
            print(f"  Batch time: {batch_elapsed:.2f}s, Archive size: {len(batch_result['archive'])}, Transfer prob: {self.transfer_probability:.3f}")
        
        total_elapsed = time.time() - start_time
        
        # Collect non-dominated set
        non_dominated_solutions = []
        if self.global_archive:
            fronts = self.fast_non_dominated_sort(self.global_archive)
            if fronts:
                non_dominated_solutions = fronts[0]
        
        # Output summary
        print(f"\nOptimization complete! Total time: {total_elapsed:.2f}s")
        print(f"Final non-dominated size: {len(non_dominated_solutions)}")
        print(f"Knowledge base size: {len(self.knowledge_base)}")
        print(f"Dynamic change count: {self.dynamic_change_count}")
        print("\nNon-dominated objectives (tardiness, waiting CV, total waiting):")
        for i, sol in enumerate(non_dominated_solutions[:10]):  # Show top 10
            print(f"Sol {i+1}: {sol.objectives}")
        
        return {
            'pareto_front': [sol.objectives for sol in non_dominated_solutions],
            'global_schedule': global_schedule,
            'batch_results': batch_results,
            'total_elapsed_time': total_elapsed,
            'knowledge_base_size': len(self.knowledge_base),
            'dynamic_change_count': self.dynamic_change_count,
            'transfer_probability': self.transfer_probability
        }

    def visualize_results(self, result: Dict):
        """Visualize results."""
        pareto_front = result.get('pareto_front', [])
        
        if not pareto_front:
            print("No solutions to visualize")
            return
        
        objectives = np.array(pareto_front)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D Pareto front
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                   c='green', s=50, alpha=0.7)
        ax1.set_xlabel('Total Tardiness')
        ax1.set_ylabel('CV of Waiting Time')
        ax1.set_zlabel('Total Waiting Time')
        ax1.set_title('Pareto Front (Tr-DMOEA) - 3D View')
        
        # 2D projection 1
        ax2 = fig.add_subplot(132)
        ax2.scatter(objectives[:, 0], objectives[:, 1], c='blue', alpha=0.7)
        ax2.set_xlabel('Total Tardiness')
        ax2.set_ylabel('CV of Waiting Time')
        ax2.set_title('Total Tardiness vs CV of Waiting Time')
        
        # 2D projection 2
        ax3 = fig.add_subplot(133)
        ax3.scatter(objectives[:, 0], objectives[:, 2], c='red', alpha=0.7)
        ax3.set_xlabel('Total Tardiness')
        ax3.set_ylabel('Total Waiting Time')
        ax3.set_title('Total Tardiness vs Total Waiting Time')
        
        plt.tight_layout()
        plt.show()

# -------------------------- Test code --------------------------
if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    random.seed(42)
    
    # Init problem instance (50 jobs, 5 machines)
    problem = JobShopProblem(num_jobs=50, num_machines=5)
    
    # Define dynamic events
    dynamic_events = [
        {
            'type': 'machine_breakdown',
            'machine_id': 1,
            'time': 30,
            'end_time': 45
        }
    ]
    
    # Create Tr-DMOEA scheduler
    print("Initializing Tr-DMOEA scheduler...")
    scheduler = Tr_DMOEAScheduler(
        problem=problem,
        dynamic_events=dynamic_events,
        pop_size=100,
        max_gen_per_batch=50,
        archive_size=50,
        transfer_pool_size=20,
        prediction_horizon=3,
        batch_size=10
    )
    
    # Run algorithm
    print("\nRunning Tr-DMOEA...")
    result = scheduler.run()
    
    # Visualize results
    scheduler.visualize_results(result)
    
    # Finish
    print("\nRun complete!")