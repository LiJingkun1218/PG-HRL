import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from dataclasses import dataclass
from common.cfunctions import (decode_schedule,evaluate_objectives)
@dataclass
class Solution:
    """Solution data structure."""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Dict = None

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

class FCPPredictor:
    """FCP predictor: analyze history to forecast next batch focus."""
    
    def __init__(self, memory_size=3):
        self.memory_size = memory_size
        self.history_features = []  # History front features
        self.history_centers = []   # History front centers
        
    def _extract_front_features(self, archive: List[Solution]) -> Dict:
        """Extract front features from archive."""
        if not archive:
            return None
            
        objectives = np.array([sol.objectives for sol in archive])
        
        # Front center: mean objectives
        center = np.mean(objectives, axis=0).tolist()
        
        # Front range: max-min per objective
        front_range = (np.max(objectives, axis=0) - np.min(objectives, axis=0)).tolist()
        
        # Front shape: inter-objective correlations
        correlations = [0, 0, 0]
        if len(objectives) > 2:
            corr_matrix = np.corrcoef(objectives.T)
            # Extract lower triangle correlations
            correlations = [
                corr_matrix[1, 0],
                corr_matrix[2, 0],
                corr_matrix[2, 1]
            ]
        
        # Front density: average crowding
        avg_density = 1.0
        if len(archive) > 2:
            crowding_distances = self._calculate_crowding_distance(archive)
            avg_density = np.mean(crowding_distances)
        
        return {
            'center': center,
            'range': front_range,
            'correlations': correlations,
            'density': avg_density,
            'size': len(archive)
        }
    
    def _calculate_crowding_distance(self, solutions: List[Solution]) -> List[float]:
        """Crowding distance (shared logic)."""
        if len(solutions) <= 2:
            return [float('inf')] * len(solutions)
        
        n = len(solutions)
        distances = [0.0] * n
        
        for m in range(3):  # Three objectives
            sorted_indices = sorted(range(n), key=lambda i: solutions[i].objectives[m])
            
            # Boundary solutions
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            min_obj = solutions[sorted_indices[0]].objectives[m]
            max_obj = solutions[sorted_indices[-1]].objectives[m]
            obj_range = max_obj - min_obj if max_obj > min_obj else 1.0
            
            for i in range(1, n - 1):
                prev_obj = solutions[sorted_indices[i-1]].objectives[m]
                next_obj = solutions[sorted_indices[i+1]].objectives[m]
                distances[sorted_indices[i]] += (next_obj - prev_obj) / obj_range
        
        return distances
    
    def update_history(self, archive: List[Solution]):
        """Update history."""
        features = self._extract_front_features(archive)
        if features:
            self.history_features.append(features)
            self.history_centers.append(features['center'])
            
            # Enforce memory size
            if len(self.history_features) > self.memory_size:
                self.history_features.pop(0)
                self.history_centers.pop(0)
    
    def predict_next_batch(self) -> Dict:
        """Predict next batch focus (multi-feature)."""
        if len(self.history_features) < 2:
            return {
                'focus_target': None,
                'predicted_center': None,
                'search_bias': {'exploration': 0.7, 'exploitation': 0.3}
            }
        
        # 1. Multi-dimensional feature analysis
        recent_centers = np.array(self.history_centers[-2:])  # Last two centers
        center_change = np.abs(recent_centers[1] - recent_centers[0])
        range_change = np.abs(np.array(self.history_features[-1]['range']) - np.array(self.history_features[-2]['range']))
        
        # Combined change score (center + range)
        total_change = center_change * 0.7 + range_change * 0.3
        most_changing_idx = np.argmax(total_change)
        target_names = ['total_tardiness', 'cv_wait_time', 'total_waiting_time']
        focus_target = target_names[most_changing_idx]
        
        # 2. Nonlinear center prediction
        predicted_center = []
        centers_array = np.array(self.history_centers)
        time_steps = np.arange(len(centers_array)).reshape(-1, 1)
        for obj_idx in range(3):
            obj_values = centers_array[:, obj_idx]
            if len(obj_values) > 1:
                # 2nd-order polynomial fit
                coeffs = np.polyfit(time_steps.flatten(), obj_values, 2)
                predicted = np.polyval(coeffs, len(centers_array))  # Next step
                predicted_center.append(predicted)
            else:
                predicted_center.append(obj_values[0])
        
        # 3. Adjust search bias by correlation and density
        recent_corr = self.history_features[-1]['correlations']
        recent_density = self.history_features[-1]['density']
        
        # More negative correlations -> exploration; higher density -> exploitation
        neg_corr_count = sum(1 for c in recent_corr if c < -0.3)
        exploration_bias = 0.5 + neg_corr_count * 0.1 + (1.0 / recent_density) * 0.1
        exploration_bias = max(0.3, min(0.8, exploration_bias))  # Clamp range
        exploitation_bias = 1.0 - exploration_bias
        
        return {
            'focus_target': focus_target,
            'predicted_center': predicted_center,
            'search_bias': {
                'exploration': exploration_bias,
                'exploitation': exploitation_bias
            },
            'correlations': recent_corr
        }

class FCPScheduler:
    """FCP-based multi-objective job shop scheduler."""
    
    # Constants
    ELITE_RATIO = 0.2
    PREDICTION_BIAS_RATIO = 0.6
    RANDOM_RATIO = 0.2
    MUT_SWAP = 0.6
    MUT_REVERSE = 0.2
    MUT_INSERT = 0.2
    TOURNAMENT_RATIO = 0.7
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=50,
                 mutation_prob=0.1, archive_size=50, batch_size=10, cache_size=500):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen_per_batch = max_gen
        self.mutation_prob = mutation_prob
        self.archive_size = archive_size
        self.batch_size = batch_size
        
        # FCP-specific components
        self.predictor = FCPPredictor(memory_size=3)
        self.global_archive = []  # Global non-dominated archive
        self.batch_archives = []  # Per-batch non-dominated sets
        self.batch_schedules = []  # Per-batch schedules
        
        # Cache config (LRU-like)
        self.cache_size = cache_size
        self._decode_cache_dict = {}
    
    def _decode_schedule_core(self, individual: List[int], problem_instance: JobShopProblem) -> Dict:
        """Key: decode core logic (for caching)."""
        # Problem instance params
        num_jobs = problem_instance.num_jobs
        num_machines = problem_instance.num_machines
        jobs_data = problem_instance.jobs_data
        
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
   
    def dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if solution 1 dominates solution 2."""
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better
    
    def fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sort (optimized indexing)."""
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
    
    def crowding_distance_assignment(self, front: List[Solution]) -> List[float]:
        """Crowding distance (shared core)."""
        return self.predictor._calculate_crowding_distance(front)
    
    def update_archive(self, archive: List[Solution], new_solutions: List[Solution]) -> List[Solution]:
        """Update archive with non-dominated solutions."""
        # Merge solutions
        all_solutions = archive + new_solutions
        
        # Non-dominated sort
        fronts = self.fast_non_dominated_sort(all_solutions)
        
        # Select non-dominated solutions
        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.archive_size:
                new_archive.extend(front)
            else:
                # Use crowding distance for last front
                remaining = self.archive_size - len(new_archive)
                if remaining > 0:
                    distances = self.crowding_distance_assignment(front)
                    sorted_indices = np.argsort(distances)[::-1]
                    for i in range(remaining):
                        new_archive.append(front[sorted_indices[i]])
                break
        
        return new_archive
    
    def get_non_dominated_set(self, solutions: List[Solution]) -> List[Solution]:
        """Extract non-dominated subset."""
        fronts = self.fast_non_dominated_sort(solutions)
        return fronts[0] if fronts else []
    
    def create_individual(self, problem_instance: JobShopProblem) -> List[int]:
        """Create individual (operation-based encoding)."""
        individual = []
        for job_id in range(problem_instance.num_jobs):
            individual.extend([job_id] * problem_instance.num_machines)
        random.shuffle(individual)
        return individual
    
    def _initialize_population_with_guidance(self, problem_instance: JobShopProblem, 
                                           prediction: Dict,
                                           previous_archive: List[Solution]) -> List[List[int]]:
        """Initialize population guided by prediction."""
        population = []
        pop_size = self.pop_size
        
        # 1. Elite guidance (20%)
        elite_count = min(int(pop_size * self.ELITE_RATIO), len(previous_archive))
        if previous_archive and elite_count > 0:
            for i in range(elite_count):
                elite_sol = previous_archive[i % len(previous_archive)]
                guided_ind = self._create_guided_individual(problem_instance, elite_sol.individual, prediction)
                population.append(guided_ind)
        
        # 2. Prediction bias (60%)
        prediction_bias_count = int(pop_size * self.PREDICTION_BIAS_RATIO)
        for _ in range(prediction_bias_count):
            base_ind = self.create_individual(problem_instance)
            biased_ind = self._bias_individual_by_prediction(base_ind, problem_instance, prediction)
            population.append(biased_ind)
        
        # 3. Random exploration (20%)
        random_count = pop_size - len(population)
        for _ in range(random_count):
            population.append(self.create_individual(problem_instance))
        
        return population
    
    def _create_guided_individual(self, problem_instance: JobShopProblem, elite_individual: List[int], 
                                prediction: Dict) -> List[int]:
        """Create guided individual from elite solution."""
        # Extract job frequency from elite
        job_freq = defaultdict(int)
        for job_id in elite_individual[:len(elite_individual)//2]:  # Use first half
            job_freq[job_id] += 1
        
        # Prioritize high-frequency jobs
        new_ind = self.create_individual(problem_instance)
        job_positions = defaultdict(list)
        for pos, job_id in enumerate(new_ind):
            job_positions[job_id].append(pos)
        
        # Sort jobs by elite frequency
        sorted_jobs = sorted(range(problem_instance.num_jobs), 
                           key=lambda x: job_freq.get(x, 0), reverse=True)
        
        # Reorder: high-frequency first
        current_pos = 0
        for job_id in sorted_jobs:
            if job_id in job_positions and current_pos < len(new_ind):
                for pos in job_positions[job_id]:
                    if pos != current_pos:
                        new_ind[current_pos], new_ind[pos] = new_ind[pos], new_ind[current_pos]
                    current_pos += 1
                    if current_pos >= len(new_ind):
                        break
        
        return new_ind
    
    def _bias_individual_by_prediction(self, individual: List[int], 
                                     problem_instance: JobShopProblem,
                                     prediction: Dict) -> List[int]:
        """Bias individual based on prediction."""
        focus_target = prediction.get('focus_target')
        
        if focus_target == 'total_tardiness':
            return self._favor_urgent_jobs(individual, problem_instance)
        elif focus_target == 'cv_wait_time':
            return self._favor_balanced_operations(individual, problem_instance)
        elif focus_target == 'total_waiting_time':
            return self._favor_short_jobs(individual, problem_instance)
        else:
            return individual
    
    def _favor_urgent_jobs(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize urgent jobs (tight due dates)."""
        job_data = problem_instance.jobs_data
        # Urgency: (due - arrival) / total processing (lower is tighter)
        job_urgency = {
            job_id: (job_data[job_id]['due_date'] - job_data[job_id]['arrival_time']) / job_data[job_id]['total_processing_time']
            for job_id in range(problem_instance.num_jobs)
        }
        
        # Collect positions and sort
        job_positions = defaultdict(list)
        for pos, job_id in enumerate(individual):
            job_positions[job_id].append(pos)
        
        sorted_jobs = sorted(range(problem_instance.num_jobs), key=lambda x: job_urgency[x])
        
        # Reorder
        new_ind = individual[:]
        current_pos = 0
        for job_id in sorted_jobs:
            if job_id in job_positions and current_pos < len(new_ind):
                for pos in job_positions[job_id]:
                    if pos != current_pos:
                        new_ind[current_pos], new_ind[pos] = new_ind[pos], new_ind[current_pos]
                    current_pos += 1
        
        return new_ind
    
    def _favor_balanced_operations(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize balanced processing time jobs (lower CV)."""
        # CV of operation times (lower is more balanced)
        job_op_cv = {}
        for job_id in range(problem_instance.num_jobs):
            op_times = problem_instance.jobs_data[job_id]['processing_times']
            if len(op_times) >= 2:
                mean = np.mean(op_times)
                if mean > 0:
                    cv = np.std(op_times, ddof=1) / mean
                    job_op_cv[job_id] = cv
                else:
                    job_op_cv[job_id] = float('inf')
            else:
                job_op_cv[job_id] = 0
        
        # Reorder
        job_positions = defaultdict(list)
        for pos, job_id in enumerate(individual):
            job_positions[job_id].append(pos)
        
        sorted_jobs = sorted(range(problem_instance.num_jobs), key=lambda x: job_op_cv.get(x, float('inf')))
        new_ind = individual[:]
        current_pos = 0
        
        for job_id in sorted_jobs:
            if job_id in job_positions and current_pos < len(new_ind):
                for pos in job_positions[job_id]:
                    if pos != current_pos:
                        new_ind[current_pos], new_ind[pos] = new_ind[pos], new_ind[current_pos]
                    current_pos += 1
        
        return new_ind
    
    def _favor_short_jobs(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize short total processing time jobs."""
        job_data = problem_instance.jobs_data
        sorted_jobs = sorted(range(problem_instance.num_jobs), 
                           key=lambda i: job_data[i]['total_processing_time'])
        
        job_positions = defaultdict(list)
        for pos, job_id in enumerate(individual):
            job_positions[job_id].append(pos)
        
        new_ind = individual[:]
        current_pos = 0
        
        for job_id in sorted_jobs:
            if job_id in job_positions and current_pos < len(new_ind):
                for pos in job_positions[job_id]:
                    if pos != current_pos:
                        new_ind[current_pos], new_ind[pos] = new_ind[pos], new_ind[current_pos]
                    current_pos += 1
        
        return new_ind
    
    def _mutate_individual(self, individual: List[int]) -> List[int]:
        """Mutation operator."""
        if random.random() > self.mutation_prob:
            return individual
        
        mutated = individual[:]
        mutation_type = random.random()
        
        if mutation_type < self.MUT_SWAP:  # 60% swap
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        elif mutation_type < self.MUT_SWAP + self.MUT_REVERSE:  # 20% reverse
            idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))
            mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])
        else:  # 20% insert
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            gene = mutated.pop(idx1)
            mutated.insert(idx2, gene)
        
        return mutated
    
    def _evolve_batch(self, problem_instance: JobShopProblem, initial_population=None) -> Dict:
        """Key: evolve one batch with non-dominated focus."""
        # Initialize population
        if initial_population:
            population = initial_population
        else:
            population = [self.create_individual(problem_instance) for _ in range(self.pop_size)]
        
        # Evaluate initial population
        solutions = []
        for ind in population:
            schedule = decode_schedule(self,ind)
            objectives = evaluate_objectives(self,schedule)
            solutions.append(Solution(ind, objectives, schedule))
        
        # Initial archive (non-dominated)
        archive = self.update_archive([], solutions)
        
        # Evolution loop
        for generation in range(self.max_gen_per_batch):
            new_solutions = []
            
            # Parent selection and offspring
            for _ in range(self.pop_size):
                # Tournament selection (prefer non-dominated)
                candidates = random.sample(solutions, min(5, len(solutions)))
                non_dominated_candidates = self.get_non_dominated_set(candidates)
                
                if non_dominated_candidates:
                    parent = random.choice(non_dominated_candidates)
                else:
                    parent = random.choice(candidates)
                
                # Mutation
                child_ind = self._mutate_individual(parent.individual[:])
                child_schedule = decode_schedule(self,child_ind)
                child_obj = evaluate_objectives(self,child_schedule)
                child_sol = Solution(child_ind, child_obj, child_schedule)
                new_solutions.append(child_sol)
            
            # Merge populations
            all_solutions = solutions + new_solutions
            
            # Update archive
            archive = self.update_archive(archive, all_solutions)
            
            # Select next generation
            fronts = self.fast_non_dominated_sort(all_solutions)
            next_solutions = []
            remaining = self.pop_size
            
            for front in fronts:
                if len(front) <= remaining:
                    next_solutions.extend(front)
                    remaining -= len(front)
                else:
                    distances = self.crowding_distance_assignment(front)
                    sorted_indices = np.argsort(distances)[::-1]
                    for i in range(remaining):
                        next_solutions.append(front[sorted_indices[i]])
                    break
            
            solutions = next_solutions
        
        # Extract batch non-dominated set
        non_dominated_set = self.get_non_dominated_set(archive)
        
        return {
            'archive': archive,
            'non_dominated_set': non_dominated_set,
            'final_population': solutions,
            'best_solutions': non_dominated_set  # Batch-best non-dominated
        }
    
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
    
    def export_non_dominated_solutions(self, solutions: List[Solution], export_path: str = None) -> Dict:
        """Export non-dominated solutions."""
        export_data = {
            'solution_count': len(solutions),
            'objectives': [sol.objectives for sol in solutions],
            'individuals': [sol.individual for sol in solutions],
            'schedule_summary': []
        }
        
        # Schedule summaries
        for sol in solutions:
            if sol.schedule and sol.schedule['valid']:
                summary = {
                    'job_completion_times': sol.schedule['job_completion_times'],
                    'makespan': max(sol.schedule['job_completion_times']),
                    'total_tardiness': sol.objectives[0],
                    'cv_wait_time': sol.objectives[1],
                    'total_waiting_time': sol.objectives[2]
                }
                export_data['schedule_summary'].append(summary)
        
        # Save to file if path provided
        if export_path:
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=4)
        
        return export_data
    
    def run(self) -> Dict:
        """Run FCP: batch optimization with cross-batch knowledge."""
        start_time = time.time()
        
        print("Starting FCP batch optimization...")
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
            
            # Prediction (none for first batch)
            prediction = self.predictor.predict_next_batch() if batch_idx > 0 else None
            if prediction:
                print(f"  Predicted focus: {prediction['focus_target']}, Exploration: {prediction['search_bias']['exploration']:.2f}")
            
            # Initialize population
            initial_pop = None
            if prediction:
                initial_pop = self._initialize_population_with_guidance(batch_problem, prediction, self.global_archive)
            
            # Batch optimization
            batch_start_time = time.time()
            batch_result = self._evolve_batch(batch_problem, initial_pop)
            batch_elapsed = time.time() - batch_start_time
            
            # Record batch non-dominated set
            self.batch_archives.append(batch_result['non_dominated_set'])
            
            # Update global archive
            self.global_archive = self.update_archive(self.global_archive, batch_result['non_dominated_set'])
            
            # Update predictor history
            self.predictor.update_history(batch_result['non_dominated_set'])
            
            # Merge batch schedule to global (first feasible non-dominated)
            if batch_result['non_dominated_set']:
                best_schedule = batch_result['non_dominated_set'][0].schedule
                self._merge_batch_schedule(global_schedule, best_schedule, batch_job_indices, time_offset)
                
                # Update time offset
                batch_makespan = max(best_schedule['job_completion_times'])
                time_offset += batch_makespan
                
                print(f"  Batch time: {batch_elapsed:.2f}s, Non-dominated count: {len(batch_result['non_dominated_set'])}")
            else:
                print(f"  Batch time: {batch_elapsed:.2f}s, No valid non-dominated solutions")
            
            batch_results.append(batch_result)
        
        # Total time
        total_elapsed = time.time() - start_time
        
        # Export global non-dominated set
        global_non_dominated = self.get_non_dominated_set(self.global_archive)
        export_data = self.export_non_dominated_solutions(global_non_dominated, "global_non_dominated_solutions.json")
        
        print(f"\nOptimization complete! Total time: {total_elapsed:.2f}s")
        print(f"Global non-dominated count: {len(global_non_dominated)}")
        print(f"Global makespan: {global_schedule['makespan']}")
        
        # Objective summary for non-dominated set
        if global_non_dominated:
            objectives = np.array([sol.objectives for sol in global_non_dominated])
            print("\nGlobal non-dominated objective stats:")
            print(f"  Tardiness - min: {objectives[:,0].min()}, max: {objectives[:,0].max()}, mean: {objectives[:,0].mean():.2f}")
            print(f"  Waiting CV - min: {objectives[:,1].min():.4f}, max: {objectives[:,1].max():.4f}, mean: {objectives[:,1].mean():.4f}")
            print(f"  Total waiting - min: {objectives[:,2].min()}, max: {objectives[:,2].max()}, mean: {objectives[:,2].mean():.2f}")
        
        return {
            'global_schedule': global_schedule,
            'pareto_front': [sol.objectives for sol in global_non_dominated],  # Objectives only
            'batch_archives': self.batch_archives,
            'total_elapsed_time': total_elapsed,
            'export_data': export_data
        }

 # Test code
if __name__ == "__main__":
    # Create problem instance
    problem = JobShopProblem(num_jobs=10, num_machines=3)
    
    # Create scheduler
    scheduler = FCPScheduler(
        problem=problem,
        pop_size=50,
        max_gen_per_batch=20,
        mutation_prob=0.1,
        archive_size=30,
        batch_size=2
    )
    
    # Run optimization
    result = scheduler.run()
    
    # Visualize non-dominated set (3D objective space)
    if result['global_non_dominated_solutions']:
        objectives = np.array([sol.objectives for sol in result['global_non_dominated_solutions']])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize objectives for visualization
        norm_obj = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-6)
        
        ax.scatter(norm_obj[:,0], norm_obj[:,1], norm_obj[:,2], c='blue', s=50, alpha=0.7)
        ax.set_xlabel('Normalized tardiness')
        ax.set_ylabel('Normalized waiting CV')
        ax.set_zlabel('Normalized total waiting')
        ax.set_title('FCP global non-dominated distribution')
        plt.savefig('non_dominated_solutions_3d.png')
        plt.show()