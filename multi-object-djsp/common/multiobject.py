import numpy as np 
import os
import pickle
from scipy import stats
import torch

class MultiObjectiveManager:
    def __init__(self, num_objectives=3, max_gen=500):
       
        self.num_objectives = num_objectives
        self.max_gen = max_gen  # Max window size
        self.external_archive = []  # Pareto archive
        self.preference_samples = []  # Preference samples
        self.samples_num = 10
        self.global_preference_baseline = None
        self.update_counter = 0
           
    def evaluate_solution(self, job_creator, job_id):
        """Evaluate three objectives for one job."""
        if job_id not in job_creator.production_record:
            return None
            
        record = job_creator.production_record[job_id]
        
        # Ensure complete production record
        if record[1] is None or record[2] is None or record[3] is None:
            return None
        
        # Objective 1: delivery deviation (absolute mean)
        objective1 = record[1]        
        # Objective 2: machine available time / actual work time
        objective2 = record[2]        
        # Objective 3: flow time ratio
        objective3 = record[3] 
        
        # Validate values
        if any(np.isnan(obj) or np.isinf(obj) for obj in [objective1, objective2, objective3]):
            print(f"Warning: invalid objective values for job {job_id}")
            return None
            
        return [objective1, objective2, objective3]
  
    def is_dominated(self, new_solution, archive):
        """Check whether a solution is dominated by any in the archive."""
        for solution in archive:
            dominated = True
            for i in range(self.num_objectives):
                if solution[i] > new_solution[i]:  # Assume minimization
                    dominated = False
                    break
            if dominated:
                return True
        return False
  
    def update_archive(self, new_solution):
        """Update archive and refresh global preferences when needed."""
        if not new_solution or any(np.isnan(obj) for obj in new_solution):
            return False
        new_solution = np.array(new_solution)
        # Key: dominance checks keep the archive Pareto-optimal
        if self.is_dominated(new_solution, self.external_archive):
            return False
        # Remove solutions dominated by the new one
        self.external_archive = [sol for sol in self.external_archive 
                               if not self.is_dominated(sol, [new_solution])]
        # Add new solution
        self.external_archive.append(new_solution)
        # Update counter
        self.update_counter += 1
        # Build preference samples when archive is large enough
        if len(self.external_archive) >= 3:  # Require at least 3 solutions
            # 1) Normalize objectives
            normalized_archive = self._normalize_objectives()
            # 2) Fit distributions
            distributions = self._fit_objective_distributions(normalized_archive)
            # 3) Preference samples
            self.preference_samples = self._generate_mixed_samples(distributions, num_samples=self.samples_num)
        
        # Periodic status
        if self.update_counter % 2 == 0 :
            print(f"  Archive size: {len(self.external_archive)}")      
        # Update baseline reference points
        self._update_preference_baseline()
        
        return True
      
    def _update_preference_baseline(self):
        """Update baseline reference points (ideal/nadir/mean)."""
        if len(self.external_archive) == 0:
            return
        # All objectives
        objectives_array = self.external_archive
        if len(objectives_array) == 0:
            return
        ideal_point = np.min(objectives_array, axis=0)  # Ideal point
        nadir_point = np.max(objectives_array, axis=0)  # Nadir point
        mean_point = np.mean(objectives_array, axis=0)  # Mean point
        self.global_preference_baseline = {
            'ideal_point': ideal_point,
            'nadir_point': nadir_point,
            'mean_point': mean_point  
        }
  
    def _normalize_objectives(self):
        """Normalize objectives to [0,1]."""
        if len(self.external_archive) < 2:
            # Return raw values
            if self.external_archive:
                return np.array(self.external_archive)
            return np.array([])
        # All objectives
        objectives_array = np.array(self.external_archive)
        # Min/max per dimension
        min_vals = np.min(objectives_array, axis=0)
        max_vals = np.max(objectives_array, axis=0)
        ranges = max_vals - min_vals
        # Avoid division by zero
        ranges = np.where(ranges < 1e-10, 1.0, ranges)
        
        # Normalize
        normalized = (objectives_array - min_vals) / ranges
        return normalized
    
    def _fit_objective_distributions(self, normalized_archive):
        distributions = {}
        n_dims = normalized_archive.shape[1]
        
        for dim in range(n_dims):
            dim_data = normalized_archive[:, dim]           
            if len(dim_data) < 3:
                distributions[dim] = {
                    'type': 'uniform',
                    'params': [0, 1],  # [a, b]
                    'function': lambda x, a=0, b=1: stats.uniform.pdf(x, a, b-a)
                }
                continue
            # Try multiple distribution fits
            best_dist = self._fit_best_distribution(dim_data)
            distributions[dim] = best_dist
            
        return distributions
    
    def _fit_best_distribution(self, data):
        """Fit the best distribution for data."""
        # Candidate distributions
        distributions_to_try = [
            ('normal', stats.norm),
            ('beta', stats.beta),
            ('gamma', stats.gamma),
            ('lognorm', stats.lognorm),
            ('uniform', stats.uniform)
        ]
        
        best_dist_name = 'uniform'
        best_dist_params = [0, 1]
        best_aic = np.inf
        best_function = lambda x: stats.uniform.pdf(x, 0, 1)
        
        for dist_name, dist_class in distributions_to_try:
            try:
                # Skip unsuitable distributions
                if dist_name == 'uniform':
                    continue  # Uniform as fallback
                    
                # Fit parameters
                params = dist_class.fit(data)
                # AIC
                log_likelihood = np.sum(dist_class.logpdf(data, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                # Update best
                if aic < best_aic and not np.isnan(aic):
                    best_aic = aic
                    best_dist_name = dist_name
                    best_dist_params = params
                    best_function = lambda x, params=params: dist_class.pdf(x, *params)
                    
            except Exception as e:
                continue
        
        return {
            'type': best_dist_name,
            'params': best_dist_params,
            'function': best_function
        }
  
    def _generate_mixed_samples(self, distributions, num_samples):
        """Mixed sampling: uniform + fitted distributions."""
        n_dims = len(distributions)
        samples = []
        # 50% uniform
        num_uniform = num_samples // 2
        uniform_samples = self._generate_uniform_samples(n_dims, num_uniform)
        samples.extend(uniform_samples)
        # 50% distribution-based
        num_dist = num_samples - num_uniform
        dist_samples = self._generate_distribution_based_samples(distributions, num_dist)
        samples.extend(dist_samples)
        return samples
   
    def _generate_uniform_samples(self, n_dims, num_samples):
        """Generate uniform preference samples."""
        samples = []
        for _ in range(num_samples):
            # Random vector normalized
            sample = np.random.rand(n_dims)
            sample = sample / np.sum(sample)
            samples.append(sample)
        return samples
   
    def _generate_distribution_based_samples(self, distributions, num_samples):
        """
        Generate preference samples from archive distributions.
        Strategy: sample each dimension then normalize.
        """
        samples = []
        n_dims = len(distributions)
        
        for _ in range(num_samples):
            sample = np.zeros(n_dims)
            
            for dim in range(n_dims):
                dist_info = distributions[dim]
                
                if dist_info['type'] == 'uniform':
                    # Uniform sample
                    sample[dim] = np.random.uniform(0, 1)
                elif dist_info['type'] == 'kde' or dist_info['type'] == 'empirical':
                    # Empirical or KDE sampling
                    if dist_info['type'] == 'kde':
                        # KDE rejection sampling
                        sample[dim] = self._rejection_sampling(dist_info['function'])
                    else:
                        # Empirical resampling
                        sample[dim] = np.random.choice(dist_info['params'])
                else:
                    # Parametric distribution sample
                    try:
                        if dist_info['type'] == 'normal':
                            # Truncate to [0,1]
                            val = np.random.normal(*dist_info['params'][:2])
                            sample[dim] = np.clip(val, 0, 1)
                        elif dist_info['type'] == 'beta':
                            sample[dim] = np.random.beta(*dist_info['params'])
                        elif dist_info['type'] == 'gamma':
                            # Scaled gamma
                            val = np.random.gamma(*dist_info['params'])
                            sample[dim] = np.clip(val / 10.0, 0, 1)
                        else:
                            # Default uniform
                            sample[dim] = np.random.uniform(0, 1)
                    except:
                        sample[dim] = np.random.uniform(0, 1)
            
            # Normalize to preference vector
            if np.sum(sample) > 0:
                sample = sample / np.sum(sample)
            else:
                sample = np.ones(n_dims) / n_dims
            
            samples.append(sample)
        
        return samples
    
    def create_default_preference(self):       
        default_vector = np.array(np.random.dirichlet([1, 1, 1]))        
        return torch.tensor(default_vector,dtype = torch.float32)
   
    def calculate_preference(self,job_idx):
        if len(self.preference_samples) > 0:           
            index = int(job_idx / self.max_gen) % (self.samples_num-1)           
            selected_preference = self.preference_samples[index].copy()           
        else:
            selected_preference = self.create_default_preference()
        return torch.tensor(selected_preference,dtype = torch.float32)
    
    def calculate_importance(self, job_creator, job_queue):
        if job_queue is None or len(job_queue) == 0 or self.global_preference_baseline is None or len(self.global_preference_baseline) == 0:
            return np.array([np.random.dirichlet([1, 1, 1]) for _ in range(len(job_queue))])
        else:
            baseline = self.global_preference_baseline
        
        # Reference points
        ideal_point = np.array(baseline['ideal_point'])      # (3,)
        nadir_point = np.array(baseline['nadir_point'])
        # 2) Estimated objectives for the queue
        all_job_values = []        
        for job_idx in job_queue:           
            job_est = np.array([
                job_creator.objects[job_idx][1],  # Objective 1 estimate
                job_creator.objects[job_idx][2],  # Objective 2 estimate
                job_creator.objects[job_idx][3]   # Objective 3 estimate
            ])
            all_job_values.append(job_est)
        all_job_values.append(ideal_point)
        all_job_values.append(nadir_point)
        queue_matrix = np.array(all_job_values)  # Shape: [n_jobs, 3]
      
        min_gap_per_obj = np.min(queue_matrix, axis=0)  # (3,)
        max_gap_per_obj = np.max(queue_matrix, axis=0)  # (3,)
        gap_ranges = max_gap_per_obj - min_gap_per_obj
        gap_ranges = np.where(gap_ranges < 1e-10, 1.0, gap_ranges)
        importance_per_objective = (queue_matrix - min_gap_per_obj) / gap_ranges        
        job_importance = importance_per_objective[:-2, :]
            
        return job_importance
    
    def save_training_results(self, file_path):
        """Save training results (fixed baseline only)."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save payload
        save_data = {
            # Basic info
            'version': '1.0',
            'num_objectives': self.num_objectives,
            
            # Core data
            'global_preference_baseline': self.global_preference_baseline,
            
            # Archive (cap at 100)
            'external_archive': self.external_archive[:100] if len(self.external_archive) > 0 else [],
            'archive_size': len(self.external_archive),           
        }
        
        # Archive summary
        if len(self.external_archive) > 0:
            archive_array = np.array(self.external_archive)
            save_data['archive_summary'] = {
                'ideal_point': np.min(archive_array, axis=0).tolist(),
                'nadir_point': np.max(archive_array, axis=0).tolist(),
                'mean_point': np.mean(archive_array, axis=0).tolist(),
            }
        
        # Save to file
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Saved training results to: {file_path}")           
            print(f"  Pareto archive size: {len(self.external_archive)}")
            if 'archive_summary' in save_data:
                print(f"  Ideal point: {save_data['archive_summary']['ideal_point']}")
            
            return True
            
        except Exception as e:
            print(f"Save failed: {e}")
            return False
    
    def get_per_baselines(self, file_path):
   
        file_path = file_path.replace('.pt', '_preferences.pkl')
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        base_baseline = save_data.get('global_preference_baseline') 
        
        self.global_preference_baseline = base_baseline
 