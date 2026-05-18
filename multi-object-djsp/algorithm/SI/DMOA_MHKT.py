
import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from dataclasses import dataclass
from functools import lru_cache
import warnings
from common.cfunctions import (decode_schedule,evaluate_objectives)
warnings.filterwarnings('ignore')

# ===================== Core data structures =====================
@dataclass
class Solution:
    """Solution container: individual, objectives, schedule."""
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Dict = None
    batch_correlation: Dict = None  # Key: batch correlation features

# ===================== Job shop problem definition =====================
class JobShopProblem:
    """Job shop scheduling problem with dynamic parameter updates."""
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
        """Generate a random instance (supports dynamic adjustments)."""
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
                'total_processing_time': sum(processing_times),
                'op_dependency': random.uniform(0.3, 0.8)  # Key: operation dependency
            })
        return jobs_data
    
    def update_job_dynamic_params(self, job_id: int, new_processing_times: List[int], new_due_date: int):
        """Dynamically update job parameters (dynamic scenario)."""
        if job_id < self.num_jobs:
            self.jobs_data[job_id]['processing_times'] = new_processing_times
            self.jobs_data[job_id]['due_date'] = new_due_date
            self.jobs_data[job_id]['total_processing_time'] = sum(new_processing_times)
    
    def calculate_batch_correlation(self) -> Dict:
        """Key: batch correlation features (local enhancement)."""
        machine_seqs = [job['machine_sequence'] for job in self.jobs_data]
        process_times = [job['processing_times'] for job in self.jobs_data]
        
        # Machine sequence overlap rate
        seq_overlap = 0
        for i in range(len(machine_seqs)):
            for j in range(i+1, len(machine_seqs)):
                overlap = len(set(machine_seqs[i]) & set(machine_seqs[j])) / self.num_machines
                seq_overlap += overlap
        seq_overlap /= max(1, len(machine_seqs)*(len(machine_seqs)-1)/2)
        
        # Processing time coefficient of variation
        all_times = [t for sublist in process_times for t in sublist]
        time_cv = np.std(all_times) / np.mean(all_times) if np.mean(all_times) > 0 else 0
        
        # Average operation dependency
        avg_op_dependency = np.mean([job['op_dependency'] for job in self.jobs_data])
        
        return {
            'seq_overlap_rate': seq_overlap,
            'process_time_cv': time_cv,
            'avg_op_dependency': avg_op_dependency
        }

# ===================== DMOA-MHKT core: environment sensing =====================
class EnvironmentMonitor:
    """Environment monitor: ECD and phase similarity (with local ECD)."""
    def __init__(self):
        self.history_environment = []  # Historical environment features
        self.batch_correlation_history = []  # Batch correlation history
    
    def extract_environment_features(self, batch_problem: JobShopProblem, batch_solutions: List[Solution]) -> Dict:
        """Extract batch environment features (problem/solutions/strategy)."""
        # 1. Problem features (jobs/machines)
        job_features = {
            'avg_processing_time': np.mean([sum(job['processing_times']) for job in batch_problem.jobs_data]),
            'due_date_range': np.ptp([job['due_date'] for job in batch_problem.jobs_data]),
            'machine_load_var': np.var([len(seq) for seq in [job['machine_sequence'] for job in batch_problem.jobs_data]]),
            'batch_correlation': batch_problem.calculate_batch_correlation()  # Key: batch correlation
        }
        
        # 2. Solution features (Pareto front)
        if batch_solutions and len(batch_solutions) > 0:
            objs = np.array([sol.objectives for sol in batch_solutions])
            # Key: POF geometric features
            pof_curvature = self.calculate_pof_curvature(objs)
            knee_points = self.detect_knee_points(objs)
            
            front_features = {
                'obj_means': objs.mean(axis=0).tolist(),
                'obj_ranges': np.ptp(objs, axis=0).tolist(),
                'crowding_density': float(np.mean(self.calculate_crowding_distance(batch_solutions))),
                'pof_curvature': float(pof_curvature),
                'knee_point_count': int(len(knee_points))
            }
        else:
            front_features = {
                'obj_means': [0.0, 0.0, 0.0],
                'obj_ranges': [0.0, 0.0, 0.0],
                'crowding_density': 0.0,
                'pof_curvature': 0.0,
                'knee_point_count': 0
            }
        
        # 3. Strategy features (defaults, updated by scheduler)
        strategy_features = {
            'mutation_prob': 0.15,
            'crossover_ratio': 0.7,
            'exploration_bias': 0.5
        }
        
        env_features = {
            'problem': job_features,
            'front': front_features,
            'strategy': strategy_features
        }
        self.history_environment.append(env_features)
        self.batch_correlation_history.append(job_features['batch_correlation'])
        return env_features
    
    def calculate_crowding_distance(self, solutions: List[Solution]) -> List[float]:
        """Crowding distance (for feature extraction)."""
        if len(solutions) <= 2:
            return [float('inf')] * len(solutions)
        
        n = len(solutions)
        distances = [0.0] * n
        for m in range(3):  # Three objectives
            sorted_indices = sorted(range(n), key=lambda i: solutions[i].objectives[m])
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
    
    def calculate_ecd(self, current_features: Dict, last_features: Dict) -> Tuple[float, float]:
        """Key: compute global and local ECD for layered transfer."""
        if not last_features:
            return 0.0, 0.0
        
        # Global ECD
        prob_change = self._calculate_feature_change(current_features['problem'], last_features['problem'])
        front_change = self._calculate_feature_change(current_features['front'], last_features['front'])
        strategy_change = self._calculate_feature_change(current_features['strategy'], last_features['strategy'])
        global_ecd = 0.4 * prob_change + 0.4 * front_change + 0.2 * strategy_change
        global_ecd = min(global_ecd, 1.0)
        
        # Local ECD (within batch)
        local_change = self._calculate_local_change(
            current_features['problem']['batch_correlation'],
            last_features['problem']['batch_correlation']
        )
        local_ecd = min(local_change, 1.0)
        
        return global_ecd, local_ecd
    
    def calculate_similarity(self, curr_features: Dict, hist_features: Dict) -> float:
        """Similarity across phases (with batch correlation weight)."""
        prob_sim = self._calculate_feature_similarity(curr_features['problem'], hist_features['problem'])
        front_sim = self._calculate_feature_similarity(curr_features['front'], hist_features['front'])
        strategy_sim = self._calculate_feature_similarity(curr_features['strategy'], hist_features['strategy'])
        
        # Batch correlation similarity
        corr_sim = self._calculate_feature_similarity(
            curr_features['problem']['batch_correlation'],
            hist_features['problem']['batch_correlation']
        )
        
        return 0.3 * prob_sim + 0.3 * front_sim + 0.1 * strategy_sim + 0.3 * corr_sim
    
    def calculate_pof_curvature(self, objs: np.ndarray) -> float:
        """Key: compute POF curvature."""
        if len(objs) < 3:
            return 0.0
        # PCA for 1D projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        objs_1d = pca.fit_transform(objs)
        # Curvature estimate
        dx = np.diff(objs_1d, axis=0)
        ddx = np.diff(dx, axis=0)
        curvature = np.mean(np.abs(ddx) / (np.abs(dx[:-1]) + 1e-6))
        return curvature
    
    def detect_knee_points(self, objs: np.ndarray) -> List[int]:
        """Key: detect knee points."""
        knee_points = []
        if len(objs) < 3:
            return knee_points
        # Normalize objectives
        objs_norm = (objs - objs.min(axis=0)) / (objs.max(axis=0) - objs.min(axis=0) + 1e-6)
        # Compute knee score per point
        for i in range(1, len(objs_norm)-1):
            prev = objs_norm[i-1]
            curr = objs_norm[i]
            next_p = objs_norm[i+1]
            # Angle between segments
            vec1 = curr - prev
            vec2 = next_p - curr
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            if cos_angle < 0.3:  # Angle < 72 degrees -> knee
                knee_points.append(i)
        return knee_points
    
    def _calculate_feature_change(self, curr: Dict, last: Dict) -> float:
        """Per-dimension change rate (simplified)."""
        changes = []
        
        for key in curr.keys():
            if key in last:
                curr_val = curr[key]
                last_val = last[key]
                
                # Dict: recurse
                if isinstance(curr_val, dict) and isinstance(last_val, dict):
                    changes.append(self._calculate_feature_change(curr_val, last_val))
                # List: element-wise
                elif isinstance(curr_val, list) and isinstance(last_val, list):
                    min_len = min(len(curr_val), len(last_val))
                    if min_len > 0:
                        for i in range(min_len):
                            try:
                                curr_elem = float(curr_val[i])
                                last_elem = float(last_val[i])
                                if last_elem != 0:
                                    change = abs(curr_elem - last_elem) / (abs(last_elem) + 1e-6)
                                    changes.append(change)
                            except (ValueError, TypeError, IndexError):
                                pass
                # Scalar
                else:
                    try:
                        curr_float = float(curr_val)
                        last_float = float(last_val)
                        if last_float != 0:
                            change = abs(curr_float - last_float) / (abs(last_float) + 1e-6)
                            changes.append(change)
                    except (ValueError, TypeError):
                        pass
        
        # Return mean change, or 0
        return np.mean(changes) if changes else 0.0
    
    def _calculate_local_change(self, curr_corr: Dict, last_corr: Dict) -> float:
        """Local change rate for batch correlation."""
        curr_vals = np.array(list(curr_corr.values()))
        last_vals = np.array(list(last_corr.values()))
        denom = np.abs(last_vals) + 1e-6
        return np.mean(np.abs(curr_vals - last_vals) / denom)
    
    def _calculate_feature_similarity(self, curr: Dict, hist: Dict) -> float:
        """Compute feature similarity."""
        curr_vals = []
        hist_vals = []
        
        for key in curr.keys():
            if key in hist:
                curr_val = curr[key]
                hist_val = hist[key]
                
                # Nested dict
                if isinstance(curr_val, dict) and isinstance(hist_val, dict):
                    # Recurse
                    curr_vals.extend(self._flatten_dict(curr_val))
                    hist_vals.extend(self._flatten_dict(hist_val))
                # List
                elif isinstance(curr_val, list) and isinstance(hist_val, list):
                    min_len = min(len(curr_val), len(hist_val))
                    for i in range(min_len):
                        try:
                            curr_vals.append(float(curr_val[i]))
                            hist_vals.append(float(hist_val[i]))
                        except (ValueError, TypeError, IndexError):
                            pass
                # Scalar
                else:
                    try:
                        curr_vals.append(float(curr_val))
                        hist_vals.append(float(hist_val))
                    except (ValueError, TypeError):
                        pass
        
        # Ensure data exists
        if not curr_vals or not hist_vals:
            return 0.0
        
        # To numpy
        curr_arr = np.array(curr_vals)
        hist_arr = np.array(hist_vals)
        
        # Normalize
        curr_norm = curr_arr / (np.linalg.norm(curr_arr) + 1e-6)
        hist_norm = hist_arr / (np.linalg.norm(hist_arr) + 1e-6)
        
        # Cosine similarity
        similarity = np.dot(curr_norm, hist_norm)
        return max(float(similarity), 0.0)
    
    def _flatten_dict(self, d: Dict) -> List[float]:
        """Flatten dict to numeric list."""
        result = []
        for value in d.values():
            if isinstance(value, dict):
                result.extend(self._flatten_dict(value))
            elif isinstance(value, list):
                for item in value:
                    try:
                        result.append(float(item))
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    result.append(float(value))
                except (ValueError, TypeError):
                    pass
        return result
# ===================== DMOA-MHKT core: multi-level knowledge transfer =====================
class MHKTManager:
    """Multi-level transfer manager: feature/strategy/solution layers."""
    def __init__(self, memory_size=5, N_h=20):
        self.memory_size = memory_size  # Memory capacity
        self.feature_memory = []  # Feature knowledge
        self.strategy_memory = []  # Strategy knowledge
        self.solution_memory = []  # Pareto front solutions
        self.N_h = N_h  # Alignment pairs
        self.knee_point_memory = []  # Knee point memory
    
    def store_knowledge(self, features: Dict, strategy: Dict, solutions: List[Solution], knee_points: List[int]):
        """Store multi-level knowledge (including knee points)."""
        # Feature layer
        self.feature_memory.append(features)
        # Strategy layer
        self.strategy_memory.append(strategy)
        # Solution layer (non-dominated only)
        non_dominated = self._select_non_dominated_solutions(solutions)
        self.solution_memory.append(non_dominated)
        # Knee point storage
        self.knee_point_memory.append(knee_points)
        
        # Enforce memory size
        if len(self.feature_memory) > self.memory_size:
            self.feature_memory.pop(0)
            self.strategy_memory.pop(0)
            self.solution_memory.pop(0)
            self.knee_point_memory.pop(0)
    
    def _select_non_dominated_solutions(self, solutions: List[Solution]) -> List[Solution]:
        """Select strictly non-dominated solutions."""
        if not solutions:
            return []
        
        # Fast non-dominated sort, keep first front
        fronts = self._fast_non_dominated_sort(solutions)
        return fronts[0] if fronts else []
    
    def transfer_knowledge(self, current_features: Dict, environment_monitor: EnvironmentMonitor) -> Dict:
        """Key: multi-level transfer using global/local ECD."""
        # Default result for safety
        default_result = {
            'features': {},
            'strategy': {'mutation_prob': 0.15, 'crossover_ratio': 0.7, 'exploration_bias': 0.5},
            'solutions': [],
            'knee_solutions': [],
            'global_ecd': 0.0,
            'local_ecd': 0.0,
            'C': 0.5
        }
        
        # Empty memory -> default
        if not self.feature_memory:
            return default_result
        
        # try:
        # 1. Global/local ECD and similarity
        last_features = self.feature_memory[-1]
        global_ecd, local_ecd = environment_monitor.calculate_ecd(current_features, last_features)
        similarities = [environment_monitor.calculate_similarity(current_features, hist_feat) 
                        for hist_feat in self.feature_memory]
        
        # Ensure valid weights
        if len(similarities) == 0 or sum(similarities) == 0:
            return {
                **default_result,
                'global_ecd': global_ecd,
                'local_ecd': local_ecd
            }
        
        weights = np.array(similarities) / (sum(similarities) + 1e-6)
        
        # 2. Compute C
        C = self._calculate_scoring_parameter(current_features, last_features)
        
        # 3. Select transfer strategy
        if global_ecd < 0.3 and local_ecd < 0.2:  # Stable global + local
            transferred_sols = self._weighted_solution_transfer(weights)
            knee_sols = self._select_knee_solutions(weights, C)
            strategy = self._weighted_strategy_transfer(weights)
            features = self._weighted_feature_transfer(weights)
        elif global_ecd < 0.7:  # Medium change
            transferred_sols = self._weighted_solution_transfer(weights, local_ecd)
            knee_sols = self._select_knee_solutions(weights, C)
            strategy = self._weighted_strategy_transfer(weights)
            hidden_features = self._learn_hidden_source(current_features)
            features = {**self._weighted_feature_transfer(weights), **hidden_features}
        else:  # High change
            transferred_sols = []
            knee_sols = []
            strategy = self._default_strategy()
            features = self._weighted_feature_transfer(weights)
        
        # Ensure required keys
        return {
            'solutions': transferred_sols or [],
            'knee_solutions': knee_sols or [],
            'strategy': strategy or default_result['strategy'],
            'features': features or default_result['features'],
            'C': C if C is not None else 0.5,
            'global_ecd': global_ecd,
            'local_ecd': local_ecd
        }
            
        # except Exception as e:
        #     print(f"[WARNING] Knowledge transfer error, using default value: {e}")
        #     return default_result  
    
    def _calculate_scoring_parameter(self, curr_features: Dict, last_features: Dict) -> float:
        """Score-based C parameter (knee ratio)."""
        # Dispersion DE
        def calculate_DE(objs: np.ndarray) -> float:
            if len(objs) == 0:
                return 0.0
            obj_mean = np.mean(objs, axis=0)
            return np.mean(np.linalg.norm(objs - obj_mean, axis=1))
        
        # DE at t-1
        last_sols = self.solution_memory[-1] if self.solution_memory else []
        last_objs = np.array([sol.objectives for sol in last_sols])
        DE_t_1 = calculate_DE(last_objs)
        
        # DE at t
        curr_objs = np.array(curr_features['front']['obj_means'])
        DE_t = calculate_DE(curr_objs.reshape(1, -1))
        
        # Compute R and C
        if min(DE_t_1, DE_t) < 1e-6:
            R = 0.0
        else:
            R = (DE_t_1 - DE_t) / min(DE_t_1, DE_t)
        C = 1 / (1 + np.exp(-R))  # Sigmoid function
        return C
    
    def _learn_hidden_source(self, current_features: Dict) -> Dict:
        """Key: MCD matrix factorization for hidden source."""
        if len(self.feature_memory) < 2:
            return {'hidden_features': None}
        
        # 1. Knowledge alignment (N_h pairs)
        K1, K2 = self._align_knowledge()
        if K1 is None or K2 is None:
            return {'hidden_features': None}
        
        # 2. MCD iterative optimization
        D = len(K1[0].individual) if K1 else 10
        W1 = np.eye(D)
        W2 = np.eye(D)
        H = np.random.rand(self.N_h, D)
        eps = 0.01  # Paper parameter
        max_iter = 50
        iter_count = 0
        
        while iter_count < max_iter:
            # Update W1 and W2
            W1 = self._update_weight_matrix(K1, H, W1)
            W2 = self._update_weight_matrix(K2, H, W2)
            
            # Update H
            H_new = self._update_hidden_source(K1, K2, H, W1, W2)
            
            # Loss
            loss = self._calculate_mcd_loss(K1, K2, H_new, W1, W2)
            if loss < eps:
                break
            
            H = H_new
            iter_count += 1
        
        # Extract hidden features
        hidden_features = {
            'hidden_means': np.mean(H, axis=0).tolist(),
            'hidden_std': np.std(H, axis=0).tolist()
        }
        return {'hidden_features': hidden_features}
    
    def _align_knowledge(self) -> Tuple[List[Solution], List[Solution]]:
        """Knowledge alignment: N_h similar pairs."""
        if len(self.solution_memory) < 2:
            return None, None
        
        # Non-dominated solutions from last two phases
        sol_set1 = self.solution_memory[-2]
        sol_set2 = self.solution_memory[-1]
        if len(sol_set1) < self.N_h or len(sol_set2) < self.N_h:
            return None, None
        
        # Pair by decision-space distance
        pairs = []
        used1 = set()
        used2 = set()
        
        for _ in range(self.N_h):
            min_dist = float('inf')
            best_i = -1
            best_j = -1
            
            for i, sol1 in enumerate(sol_set1):
                if i in used1:
                    continue
                for j, sol2 in enumerate(sol_set2):
                    if j in used2:
                        continue
                    dist = np.linalg.norm(np.array(sol1.individual) - np.array(sol2.individual))
                    if dist < min_dist:
                        min_dist = dist
                        best_i = i
                        best_j = j
            
            if best_i != -1 and best_j != -1:
                pairs.append((sol_set1[best_i], sol_set2[best_j]))
                used1.add(best_i)
                used2.add(best_j)
        
        if len(pairs) < self.N_h:
            return None, None
        
        K1 = [p[0] for p in pairs]
        K2 = [p[1] for p in pairs]
        return K1, K2
    
    def _update_weight_matrix(self, K: List[Solution], H: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Update W (paper Eq. 11)."""
        K_mat = np.array([sol.individual for sol in K])
        H_T_H = H.T @ H
        numerator = K_mat.T @ H
        denominator = H_T_H @ W + 1e-6
        return (numerator / denominator) @ W
    
    def _update_hidden_source(self, K1: List[Solution], K2: List[Solution], H: np.ndarray, 
                             W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
        """Update hidden source H (paper Eq. 12)."""
        K1_mat = np.array([sol.individual for sol in K1])
        K2_mat = np.array([sol.individual for sol in K2])
        
        numerator = K1_mat @ W1.T + K2_mat @ W2.T
        denominator = H @ W1 @ W1.T + H @ W2 @ W2.T + 1e-6
        return (numerator / denominator) @ H
    
    def _calculate_mcd_loss(self, K1: List[Solution], K2: List[Solution], H: np.ndarray, 
                           W1: np.ndarray, W2: np.ndarray) -> float:
        """Compute MCD loss (paper Eq. 13)."""
        K1_mat = np.array([sol.individual for sol in K1])
        K2_mat = np.array([sol.individual for sol in K2])
        
        loss1 = np.linalg.norm(K1_mat - H @ W1)
        loss2 = np.linalg.norm(K2_mat - H @ W2)
        return (loss1 + loss2) / 2
    
    def _weighted_solution_transfer(self, weights: np.array, local_ecd: float = 0.0) -> List[Solution]:
        """Weighted solution transfer (with local perturbation)."""
        all_solutions = []
        for i, sols in enumerate(self.solution_memory):
            if len(sols) == 0:
                continue
            # Increase weight under local perturbation
            weight = weights[i] * (1 + local_ecd)
            select_num = max(1, int(len(sols) * weight))
            all_solutions.extend(random.sample(sols, min(select_num, len(sols))))
        return all_solutions
    
    def _select_knee_solutions(self, weights: np.array, C: float) -> List[Solution]:
        """Select knee solutions using C."""
        knee_solutions = []
        for i, (sols, knees) in enumerate(zip(self.solution_memory, self.knee_point_memory)):
            if len(sols) == 0 or len(knees) == 0:
                continue
            # Select by weight and C
            select_num = max(1, int(len(knees) * weights[i] * C))
            knee_indices = random.sample(knees, min(select_num, len(knees)))
            knee_solutions.extend([sols[idx] for idx in knee_indices])
        return knee_solutions
    
    def _weighted_strategy_transfer(self, weights: np.array) -> Dict:
        """Weighted strategy transfer (safe defaults)."""
        if not self.strategy_memory:
            return {'mutation_prob': 0.15, 'crossover_ratio': 0.7, 'exploration_bias': 0.5}
        
        strategy_keys = ['mutation_prob', 'crossover_ratio', 'exploration_bias']
        transferred_strategy = {}
        
        for key in strategy_keys:
            values = []
            for i in range(min(len(self.strategy_memory), len(weights))):
                if key in self.strategy_memory[i]:
                    values.append(self.strategy_memory[i][key])
            
            if values:
                transferred_strategy[key] = np.sum(np.array(values) * weights[:len(values)])
            else:
                transferred_strategy[key] = 0.15 if key == 'mutation_prob' else 0.7 if key == 'crossover_ratio' else 0.5
        
        return transferred_strategy  
    
    def _weighted_feature_transfer(self, weights: np.array) -> Dict:
        """Weighted feature transfer."""
        if not self.feature_memory:
            return {}
        return self.feature_memory[-1]  # Default to latest feature
    
    def _default_strategy(self) -> Dict:
        """Default strategy under high change (more exploration)."""
        return {'mutation_prob': 0.2, 'crossover_ratio': 0.7, 'exploration_bias': 0.7}
    
    def _fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sort."""
        if not solutions:
            return [[]]
        
        n = len(solutions)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(solutions[i].objectives, solutions[j].objectives):
                    dominated_solutions[i].append(j)
                elif self._dominates(solutions[j].objectives, solutions[i].objectives):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(solutions[i])
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for sol in fronts[current_front]:
                idx = solutions.index(sol)
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(solutions[dominated_idx])
            if next_front:
                fronts.append(next_front)
            current_front += 1
        return fronts
    
    def _dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Dominance check."""
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better

# ===================== DMOA-MHKT scheduler core =====================
class DMOA_MHKTScheduler:
    """DMOA-MHKT dynamic multi-objective scheduler."""
    # Constants
    ELITE_RATIO = 0.2
    PREDICTION_BIAS_RATIO = 0.6
    RANDOM_RATIO = 0.2
    MUT_SWAP = 0.6
    MUT_REVERSE = 0.2
    MUT_INSERT = 0.2
    TOURNAMENT_RATIO = 0.7
    
    def __init__(self, problem: JobShopProblem, pop_size=100, max_gen=50,
                 mutation_prob=0.1, archive_size=50, batch_size=10, cache_size=500, N_h=20):
        # Core parameters
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen_per_batch = max_gen
        self.mutation_prob = mutation_prob
        self.archive_size = archive_size  # Archive size for non-dominated set
        self.batch_size = batch_size
        
        # DMOA-MHKT core components
        self.env_monitor = EnvironmentMonitor()
        self.mhkt_manager = MHKTManager(memory_size=5, N_h=N_h)
        
        # Cache config
        self.cache_size = cache_size
        self._decode_cache = self._create_cache_wrapper()
        
        # Global Pareto sets
        self.global_pareto_front = []  # Global Pareto front
        self.batch_pareto_fronts = []  # Batch Pareto fronts
    
    def _create_cache_wrapper(self):
        """Cache wrapper for decode scheduling."""
        def wrapper(func):
            @lru_cache(maxsize=self.cache_size)
            def cached_func(individual_tuple, num_jobs, num_machines, jobs_data_cache):
                return func(individual_tuple, num_jobs, num_machines, jobs_data_cache)
            return cached_func
        
        return wrapper(self._decode_schedule_core)
    
    # ===================== Decode and objective evaluation =====================
    def _prepare_jobs_data_for_cache(self, jobs_data: List[Dict]) -> Tuple:
        """Prepare jobs_data for cache (hashable)."""
        cache_data = []
        for job in jobs_data:
            # Extract key fields (hashable)
            cache_data.append((
                job.get('job_id', 0),
                job.get('arrival_time', 0),
                job.get('due_date', 100),
                tuple(job.get('machine_sequence', [])),  # List to tuple
                tuple(job.get('processing_times', [])),   # List to tuple
                job.get('total_processing_time', 0),
                job.get('op_dependency', 0.5)
            ))
        return tuple(cache_data)
    
    def _restore_jobs_data_from_cache(self, jobs_data_cache: Tuple) -> List[Dict]:
        """Restore jobs_data dict from cache."""
        jobs_data = []
        for cache_item in jobs_data_cache:
            job_dict = {
                'job_id': cache_item[0],
                'arrival_time': cache_item[1],
                'due_date': cache_item[2],
                'machine_sequence': list(cache_item[3]),
                'processing_times': list(cache_item[4]),
                'total_processing_time': cache_item[5],
                'op_dependency': cache_item[6]
            }
            jobs_data.append(job_dict)
        return jobs_data
 
    def _decode_schedule_core(self, individual_tuple: Tuple[int], num_jobs: int, num_machines: int, 
                             jobs_data_cache: Tuple) -> Dict:
        """Key: decode core logic."""
        individual = list(individual_tuple)
        
        # Validate chromosome
        if len(individual) == 0:
            return self._create_invalid_schedule(num_machines, num_jobs)
            
        for job_id in individual:
            if job_id is None or not isinstance(job_id, int) or job_id < 0 or job_id >= num_jobs:
                return self._create_invalid_schedule(num_machines, num_jobs)
        
        # Restore jobs_data
        jobs_data = self._restore_jobs_data_from_cache(jobs_data_cache)
        
        # Initialize schedules and job progress
        machine_schedules = [[] for _ in range(num_machines)]
        job_progress = [{
            'current_op': 0,
            'completion_time': jobs_data[i]['arrival_time'] if i < len(jobs_data) else 0
        } for i in range(num_jobs)]
        
        # Decode logic
        total_operations = len(individual)
        scheduled_flags = [False] * total_operations
        
        while True:
            scheduled_this_round = False
            
            for pos in range(total_operations):
                if scheduled_flags[pos]:
                    continue
                    
                job_id = individual[pos]
                if job_id >= len(jobs_data):
                    scheduled_flags[pos] = True
                    continue
                    
                job_data = jobs_data[job_id]
                op_index = job_progress[job_id]['current_op']
                if op_index >= num_machines:
                    scheduled_flags[pos] = True
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
                scheduled_flags[pos] = True
                scheduled_this_round = True
            
            # No operation scheduled -> exit
            if not scheduled_this_round:
                break
        
        return {
            'machine_schedules': machine_schedules,
            'job_completion_times': [job_progress[i]['completion_time'] for i in range(num_jobs)],
            'valid': True
        }
    
    def _create_invalid_schedule(self, num_machines: int, num_jobs: int) -> Dict:
        """Create invalid schedule."""
        return {
            'machine_schedules': [[] for _ in range(num_machines)],
            'job_completion_times': [float('inf') for _ in range(num_jobs)],
            'valid': False
        }
  
    # ===================== Non-dominated sort and archive =====================
    def dominates(self, obj1: Tuple[float, float, float], obj2: Tuple[float, float, float]) -> bool:
        """Check if solution 1 dominates solution 2."""
        all_not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        at_least_one_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        return all_not_worse and at_least_one_better
    
    def fast_non_dominated_sort(self, solutions: List[Solution]) -> List[List[Solution]]:
        """Fast non-dominated sort."""
        if not solutions:
            return []
            
        n = len(solutions)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = []
        
        # Pass 1: dominance relations
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
                if not fronts:
                    fronts.append([])
                fronts[0].append(sol_i)
        
        # Remaining fronts
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for sol in fronts[current_front]:
                idx = solutions.index(sol)
                for dominated_idx in dominated_solutions[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(solutions[dominated_idx])
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def crowding_distance_assignment(self, front: List[Solution]) -> List[float]:
        """Crowding distance."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        n = len(front)
        distances = [0.0] * n
        
        # Per objective
        for m in range(3):
            sorted_indices = sorted(range(n), key=lambda i: front[i].objectives[m])
            
            # Boundary points
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Interior points
            min_obj = front[sorted_indices[0]].objectives[m]
            max_obj = front[sorted_indices[-1]].objectives[m]
            obj_range = max_obj - min_obj if max_obj > min_obj else 1.0
            
            for i in range(1, n - 1):
                prev_obj = front[sorted_indices[i-1]].objectives[m]
                next_obj = front[sorted_indices[i+1]].objectives[m]
                distances[sorted_indices[i]] += (next_obj - prev_obj) / obj_range
        
        return distances
    
    def update_pareto_front(self, current_front: List[Solution], new_solutions: List[Solution]) -> List[Solution]:
        """Update Pareto front."""
        # Merge solutions
        all_solutions = current_front + new_solutions
        if not all_solutions:
            return []
        
        # Non-dominated sort
        fronts = self.fast_non_dominated_sort(all_solutions)
        if not fronts:
            return []
        
        pareto_front = fronts[0]
        
        # Truncate by crowding distance
        if len(pareto_front) > self.archive_size:
            distances = self.crowding_distance_assignment(pareto_front)
            # Sort by crowding distance
            sorted_front = [sol for _, sol in sorted(zip(distances, pareto_front), key=lambda x: x[0], reverse=True)]
            pareto_front = sorted_front[:self.archive_size]
        
        return pareto_front
    
    # ===================== Evolution core =====================
    def create_individual(self, problem_instance: JobShopProblem) -> List[int]:
        """Create individual."""
        individual = []
        for job_id in range(problem_instance.num_jobs):
            individual.extend([job_id] * problem_instance.num_machines)
        random.shuffle(individual)
        return individual
    
    def _mutate_individual(self, individual: List[int]) -> List[int]:
        """Mutation operator."""
        if random.random() > self.mutation_prob or len(individual) < 2:
            return individual[:]
        
        mutated = individual[:]
        mutation_type = random.random()
        
        if mutation_type < self.MUT_SWAP:  # Swap mutation
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        elif mutation_type < self.MUT_SWAP + self.MUT_REVERSE:  # Reverse mutation
            idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))
            mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])
        else:  # Insert mutation
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            gene = mutated.pop(idx1)
            mutated.insert(idx2, gene)
        
        return mutated
    
    def _evolve_batch(self, problem_instance: JobShopProblem, initial_pop=None, transferred_knowledge: Dict = None) -> Dict:
        """Key: evolve a single batch."""
        # Initialize population
        if initial_pop and len(initial_pop) > 0:
            population = initial_pop[:self.pop_size]
            # Fill with random individuals
            while len(population) < self.pop_size:
                population.append(self.create_individual(problem_instance))
        else:
            population = [self.create_individual(problem_instance) for _ in range(self.pop_size)]
        
        # Evaluate initial population
        solutions = []
        for ind in population:
            try:
                schedule = decode_schedule(self,ind)
                objectives = evaluate_objectives(self,schedule)
                batch_corr = problem_instance.calculate_batch_correlation()
                solutions.append(Solution(ind, objectives, schedule, batch_corr))
            except Exception as e:
                # Decode failed -> invalid solution
                solutions.append(Solution(ind, (float('inf'), float('inf'), float('inf'))))
        
        # Initialize batch Pareto front
        batch_pareto_front = self.update_pareto_front([], solutions)
        
        # Evolution loop
        for generation in range(self.max_gen_per_batch):
            new_solutions = []
            
            # Generate offspring
            for _ in range(self.pop_size):
                # Parent selection
                if len(batch_pareto_front) >= 2:
                    idx1, idx2 = random.sample(range(len(batch_pareto_front)), 2)
                    parent = batch_pareto_front[idx1] if random.random() < 0.5 else batch_pareto_front[idx2]
                else:
                    parent = random.choice(solutions)
                
                # Mutation
                child_ind = self._mutate_individual(parent.individual[:])
                
                # Evaluate offspring
                try:
                    child_schedule = decode_schedule(self,child_ind)
                    child_obj = evaluate_objectives(self,child_schedule)
                    child_sol = Solution(child_ind, child_obj, child_schedule)
                    new_solutions.append(child_sol)
                except:
                    # Decode failed -> skip
                    continue
            
            # Update batch Pareto front
            batch_pareto_front = self.update_pareto_front(batch_pareto_front, new_solutions)
        
        # Detect knee points
        objs = np.array([sol.objectives for sol in batch_pareto_front]) if batch_pareto_front else np.array([])
        knee_points = self.env_monitor.detect_knee_points(objs) if len(objs) > 0 else []
        
        return {
            'batch_pareto_front': batch_pareto_front,
            'all_solutions': solutions,
            'knee_points': knee_points,
            'reference_solution': random.choice(batch_pareto_front) if batch_pareto_front else None
        }
    
    # ===================== Transfer adaptation =====================
    def _adapt_historical_solutions(self, historical_solutions: List[Solution], batch_problem: JobShopProblem) -> List[List[int]]:
        """Adapt historical non-dominated solutions to current batch."""
        if not historical_solutions:
            return []
        
        adapted_pop = []
        for sol in historical_solutions:
            hist_ind = sol.individual
            # Simple mapping by modulo
            new_ind = [(job_id % batch_problem.num_jobs) for job_id in hist_ind]
            
            # Ensure correct length
            expected_len = batch_problem.num_jobs * batch_problem.num_machines
            if len(new_ind) < expected_len:
                new_ind.extend([random.randint(0, batch_problem.num_jobs-1) 
                              for _ in range(expected_len - len(new_ind))])
            elif len(new_ind) > expected_len:
                new_ind = new_ind[:expected_len]
            
            adapted_pop.append(new_ind)
        
        return adapted_pop
    
    def _initialize_population_with_knowledge(self, batch_problem: JobShopProblem, features: Dict) -> List[List[int]]:
        """Initialize population using transferred features."""
        pop_size = self.pop_size
        population = []
        
        # Feature-guided initialization
        if features and 'front' in features:
            front_means = features['front']['obj_means']
            
            if len(front_means) >= 3:
                # Choose strategy by objective means
                max_obj_idx = np.argmax(front_means[:3])
                
                for _ in range(pop_size):
                    ind = self.create_individual(batch_problem)
                    
                    if max_obj_idx == 0:  # High tardiness -> urgent jobs
                        ind = self._favor_urgent_jobs(ind, batch_problem)
                    elif max_obj_idx == 1:  # High variability -> balanced jobs
                        ind = self._favor_balanced_operations(ind, batch_problem)
                    else:  # High waiting -> short jobs
                        ind = self._favor_short_jobs(ind, batch_problem)
                    
                    population.append(ind)
            else:
                population = [self.create_individual(batch_problem) for _ in range(pop_size)]
        else:
            population = [self.create_individual(batch_problem) for _ in range(pop_size)]
        
        return population
    
    def _favor_urgent_jobs(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize urgent jobs."""
        jobs_data = problem_instance.jobs_data
        job_urgency = {}
        
        for job_id in range(problem_instance.num_jobs):
            if job_id < len(jobs_data):
                job_data = jobs_data[job_id]
                due_date = job_data['due_date']
                arrival_time = job_data['arrival_time']
                total_time = job_data['total_processing_time']
                job_urgency[job_id] = due_date - arrival_time - total_time
            else:
                job_urgency[job_id] = 0
        
        # Sort by urgency
        sorted_jobs = sorted(range(problem_instance.num_jobs), key=lambda x: job_urgency.get(x, 0))
        
        # Build new individual
        new_ind = []
        job_counts = defaultdict(int)
        
        # Count jobs
        for job_id in individual:
            if job_id < problem_instance.num_jobs:
                job_counts[job_id] += 1
        
        # Reorder by urgency
        for job_id in sorted_jobs:
            count = job_counts.get(job_id, 0)
            new_ind.extend([job_id] * count)
        
        return new_ind
    
    def _favor_balanced_operations(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize jobs with balanced processing times."""
        jobs_data = problem_instance.jobs_data
        job_cv = {}
        
        for job_id in range(problem_instance.num_jobs):
            if job_id < len(jobs_data):
                op_times = jobs_data[job_id]['processing_times']
                if len(op_times) >= 2:
                    mean = np.mean(op_times)
                    if mean > 0:
                        cv = np.std(op_times, ddof=1) / mean
                        job_cv[job_id] = cv
                    else:
                        job_cv[job_id] = 0
                else:
                    job_cv[job_id] = 0
            else:
                job_cv[job_id] = 0
        
        # Sort by balance (lower CV first)
        sorted_jobs = sorted(range(problem_instance.num_jobs), key=lambda x: job_cv.get(x, 0))
        
        # Build new individual
        new_ind = []
        job_counts = defaultdict(int)
        
        for job_id in individual:
            if job_id < problem_instance.num_jobs:
                job_counts[job_id] += 1
        
        for job_id in sorted_jobs:
            count = job_counts.get(job_id, 0)
            new_ind.extend([job_id] * count)
        
        return new_ind
    
    def _favor_short_jobs(self, individual: List[int], problem_instance: JobShopProblem) -> List[int]:
        """Prioritize short total processing time jobs."""
        jobs_data = problem_instance.jobs_data
        job_length = {}
        
        for job_id in range(problem_instance.num_jobs):
            if job_id < len(jobs_data):
                job_length[job_id] = jobs_data[job_id]['total_processing_time']
            else:
                job_length[job_id] = 0
        
        sorted_jobs = sorted(range(problem_instance.num_jobs), key=lambda x: job_length.get(x, 0))
        
        new_ind = []
        job_counts = defaultdict(int)
        
        for job_id in individual:
            if job_id < problem_instance.num_jobs:
                job_counts[job_id] += 1
        
        for job_id in sorted_jobs:
            count = job_counts.get(job_id, 0)
            new_ind.extend([job_id] * count)
        
        return new_ind
    
    # ===================== Batch processing helpers =====================
    def _create_batch_problem(self, batch_job_indices: List[int], time_offset: int) -> JobShopProblem:
        """Create batch problem instance."""
        batch_arrival = []
        batch_due = []
        batch_machine_seq = []
        batch_process_times = []
        
        for global_job_id in batch_job_indices:
            if global_job_id < len(self.problem.jobs_data):
                job_data = self.problem.jobs_data[global_job_id]
                batch_arrival.append(job_data['arrival_time'] + time_offset)
                batch_due.append(job_data['due_date'] + time_offset)
                batch_machine_seq.append(job_data['machine_sequence'])
                batch_process_times.append(job_data['processing_times'])
            else:
                # Defaults
                batch_arrival.append(time_offset)
                batch_due.append(time_offset + 100)
                batch_machine_seq.append(list(range(self.problem.num_machines)))
                batch_process_times.append([1] * self.problem.num_machines)
        
        return JobShopProblem(
            num_jobs=len(batch_job_indices),
            num_machines=self.problem.num_machines,
            arrival_time=batch_arrival,
            due_date=batch_due,
            machine_sequence=batch_machine_seq,
            processing_times=batch_process_times
        )
    
    # ===================== Main run logic =====================
    def run(self) -> Dict:
        """Run DMOA-MHKT algorithm."""
        start_time = time.time()
        
        print("=== Starting DMOA-MHKT dynamic multi-objective JSS optimization ===")
        print(f"Config - Jobs: {self.problem.num_jobs}, Batch size: {self.batch_size}, Machines: {self.problem.num_machines}")
        print(f"Algo config - Population: {self.pop_size}, Gens per batch: {self.max_gen_per_batch}")
        
        # Initialize
        all_jobs = list(range(self.problem.num_jobs))
        num_batches = (len(all_jobs) + self.batch_size - 1) // self.batch_size
        batch_metrics = []
        
        for batch_idx in range(num_batches):
            # Simulate dynamic scenario
            if batch_idx >= 2 and batch_idx < len(all_jobs):
                target_job_id = batch_idx * self.batch_size % self.problem.num_jobs
                if target_job_id < len(self.problem.jobs_data):
                    self.problem.update_job_dynamic_params(
                        job_id=target_job_id,
                        new_processing_times=[random.randint(2, 25) for _ in range(self.problem.num_machines)],
                        new_due_date=random.randint(60, 180)
                    )
            
            # Batch job indices
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(all_jobs))
            batch_job_indices = all_jobs[batch_start:batch_end]
            batch_size = len(batch_job_indices)
            
            print(f"\n=== Processing batch {batch_idx + 1}/{num_batches} ===")
            print(f"  Job range: {batch_start}-{batch_end-1} ({batch_size} jobs)")
            
            # 1. Create batch problem instance
            batch_problem = self._create_batch_problem(batch_job_indices, 0)
            
            # 2. Environment sensing
            dummy_solutions = [Solution(self.create_individual(batch_problem), (0,0,0))]
            current_env_features = self.env_monitor.extract_environment_features(batch_problem, dummy_solutions)
            
            # 3. Multi-level knowledge transfer
            transferred_knowledge = self.mhkt_manager.transfer_knowledge(current_env_features, self.env_monitor)
            global_ecd = transferred_knowledge['global_ecd']
            local_ecd = transferred_knowledge['local_ecd']
            print(f"  Environment change - Global ECD: {global_ecd:.2f}, Local ECD: {local_ecd:.2f}")
            
            # 4. Initialize population
            initial_pop = None
            if transferred_knowledge['solutions']:
                initial_pop = self._adapt_historical_solutions(transferred_knowledge['solutions'], batch_problem)
            
            # 5. Adjust strategy parameters
            if transferred_knowledge['strategy']:
                self.mutation_prob = transferred_knowledge['strategy']['mutation_prob']
            
            # 6. Batch evolution optimization
            batch_start_time = time.time()
            batch_result = self._evolve_batch(batch_problem, initial_pop, transferred_knowledge)
            batch_elapsed = time.time() - batch_start_time
            batch_pareto_front = batch_result['batch_pareto_front']
            
            # 7. Record batch results
            self.batch_pareto_fronts.append(batch_pareto_front)
            
            # 8. Update global Pareto front
            self.global_pareto_front = self.update_pareto_front(self.global_pareto_front, batch_pareto_front)
            
            # 9. Store knowledge
            current_strategy = {
                'mutation_prob': self.mutation_prob,
                'crossover_ratio': self.TOURNAMENT_RATIO,
                'exploration_bias': transferred_knowledge.get('strategy', {}).get('exploration_bias', 0.5) if transferred_knowledge and transferred_knowledge.get('strategy') else 0.5
            }
            self.mhkt_manager.store_knowledge(current_env_features, current_strategy, 
                                            batch_pareto_front, batch_result['knee_points'])
            
            # 10. Record metrics
            if batch_pareto_front:
                batch_obj_matrix = np.array([sol.objectives for sol in batch_pareto_front])
                batch_metrics.append({
                    'batch_idx': batch_idx,
                    'num_jobs': batch_size,
                    'pareto_size': len(batch_pareto_front),
                    'global_ecd': global_ecd,
                    'local_ecd': local_ecd,
                    'elapsed_time': batch_elapsed,
                    'obj_means': batch_obj_matrix.mean(axis=0).tolist(),
                    'obj_mins': batch_obj_matrix.min(axis=0).tolist()
                })
                
                print(f"  Batch time: {batch_elapsed:.2f}s, Pareto size: {len(batch_pareto_front)}")
                print(f"  Obj means - Tardiness: {batch_metrics[-1]['obj_means'][0]:.2f}, Variability: {batch_metrics[-1]['obj_means'][1]:.2f}, Waiting: {batch_metrics[-1]['obj_means'][2]:.2f}")
        
        # Summary
        total_elapsed = time.time() - start_time
        
        # Build simplified non-dominated set (objectives only)
        simplified_pareto_front = []
        if self.global_pareto_front:
            for sol in self.global_pareto_front:
                # Objectives only, no extra info
                simplified_pareto_front.append(sol.objectives)
        
        # Build simplified batch non-dominated sets
        simplified_batch_fronts = []
        for batch_front in self.batch_pareto_fronts:
            batch_simplified = []
            for sol in batch_front:
                batch_simplified.append(sol.objectives)
            simplified_batch_fronts.append(batch_simplified)
        
        # Global stats (simplified)
        global_stats = {'pareto_size': 0, 'obj_means': [float('inf')]*3}
        if self.global_pareto_front:
            global_obj_matrix = np.array([sol.objectives for sol in self.global_pareto_front])
            global_stats = {
                'pareto_size': len(self.global_pareto_front),
                'obj_means': global_obj_matrix.mean(axis=0).tolist(),
                'obj_mins': global_obj_matrix.min(axis=0).tolist(),
                'obj_maxs': global_obj_matrix.max(axis=0).tolist()
            }
        
        print("\n=== DMOA-MHKT optimization complete ===")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Global Pareto size: {global_stats['pareto_size']}")
        print(f"Global objective means: {global_stats['obj_means']}")
        
        # Return simplified structure for run_algorithm compatibility
        return {
            'pareto_front': simplified_pareto_front,  # Simplified objective list
            'batch_pareto_fronts': simplified_batch_fronts,  # Simplified batch objective lists
            'global_stats': global_stats,
            'batch_metrics': batch_metrics,
            'total_elapsed_time': total_elapsed,
            'raw_solutions': self.global_pareto_front  # Keep raw Solution objects (optional)
        }

# ===================== Visualization =====================
def visualize_pareto_front(pareto_front: List[Solution], title: str = "Pareto front"):
    """Visualize Pareto front."""
    if not pareto_front:
            print("No Pareto front available for visualization")
            return
    
    objs = np.array([sol.objectives for sol in pareto_front])
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: tardiness vs waiting time CV
    ax1 = fig.add_subplot(131)
    ax1.scatter(objs[:, 0], objs[:, 1], c='blue', alpha=0.7, s=50)
    ax1.set_xlabel('Total tardiness')
    ax1.set_ylabel('Waiting time CV sum')
    ax1.set_title(f'{title}: tardiness vs variability')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: tardiness vs total waiting time
    ax2 = fig.add_subplot(132)
    ax2.scatter(objs[:, 0], objs[:, 2], c='green', alpha=0.7, s=50)
    ax2.set_xlabel('Total tardiness')
    ax2.set_ylabel('Total waiting time')
    ax2.set_title(f'{title}: tardiness vs total waiting')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: variability vs total waiting time
    ax3 = fig.add_subplot(133)
    ax3.scatter(objs[:, 1], objs[:, 2], c='red', alpha=0.7, s=50)
    ax3.set_xlabel('Waiting time CV sum')
    ax3.set_ylabel('Total waiting time')
    ax3.set_title(f'{title}: variability vs total waiting')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Create problem instance
    problem = JobShopProblem(num_jobs=20, num_machines=4)
    
    # 2. Create scheduler
    scheduler = DMOA_MHKTScheduler(
        problem=problem,
        pop_size=80,
        max_gen_per_batch=30,
        mutation_prob=0.15,
        archive_size=40,
        batch_size=5,
        cache_size=500,
        N_h=20
    )
    
    # 3. Run algorithm
    result = scheduler.run()
    
    # 4. Visualization
    if result['pareto_front']:
        visualize_pareto_front(result['pareto_front'], title="Global Pareto front")
        
        # Visualize last batch
        if result['batch_pareto_fronts'] and result['batch_pareto_fronts'][-1]:
            visualize_pareto_front(result['batch_pareto_fronts'][-1], title="Last batch Pareto front")
    
    # 5. Output example
    print("\n=== Example objectives for first 5 global Pareto solutions ===")
    for i, sol in enumerate(result['pareto_front'][:5]):
        print(f"Sol {i+1}: {sol.objectives}")