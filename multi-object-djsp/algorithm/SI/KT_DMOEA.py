import numpy as np
import random
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common.cfunctions import (decode_schedule, evaluate_objectives)


# ============================================================================
# Data Structure
# ============================================================================
@dataclass
class Solution:
    individual: List[int]
    objectives: Tuple[float, float, float]
    schedule: Dict = field(default_factory=dict)


# ============================================================================
# Problem definition
# ============================================================================
class JobShopProblem:
    def __init__(self, num_jobs=50, num_machines=5, arrival_time=None, due_date=None,
                 machine_sequence=None, processing_times=None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.arrival_time = arrival_time if arrival_time is not None else [None] * num_jobs
        self.due_date = due_date if due_date is not None else [None] * num_jobs
        self.machine_sequence = machine_sequence if machine_sequence is not None else [[] for _ in range(num_jobs)]
        self.processing_times = processing_times if processing_times is not None else [[] for _ in range(num_jobs)]
        self.jobs_data = self._build_jobs_data()

    def _build_jobs_data(self):
        jobs_data = []
        for job_id in range(self.num_jobs):
            jobs_data.append({
                'job_id': job_id,
                'arrival_time': self.arrival_time[job_id] if self.arrival_time[job_id] is not None else 0,
                'due_date': self.due_date[job_id] if self.due_date[job_id] is not None else 0,
                'machine_sequence': self.machine_sequence[job_id],
                'processing_times': self.processing_times[job_id],
                'total_processing_time': sum(self.processing_times[job_id]) if self.processing_times[job_id] else 0
            })
        return jobs_data


# ============================================================================
# Knee point extraction
# ============================================================================
class KneePointExtractor:
    def __init__(self, part_num=10):
        self.part_num = part_num

    def get_knee_points(self, pareto_front):
        if len(pareto_front) < 1:
            return pareto_front
        min_obj = np.min(pareto_front, axis=0)
        max_obj = np.max(pareto_front, axis=0)
        range_obj = max_obj - min_obj
        range_obj[range_obj == 0] = 1e-6
        norm_front = (pareto_front - min_obj) / range_obj
        ideal = np.min(norm_front, axis=0)
        nadir = np.max(norm_front, axis=0)
        scores = []
        for p in norm_front:
            d1 = np.linalg.norm(p - ideal)
            d2 = np.linalg.norm(p - nadir)
            scores.append(d1 + d2)
        best_indices = np.argsort(scores)[-self.part_num:]
        return pareto_front[best_indices]


# ============================================================================
# TPM trend prediction
# ============================================================================
class TPMPredictor:
    def __init__(self, n_dim):
        self.n_dim = n_dim

    def polar_transform(self, v):
        n = len(v)
        if n == 1:
            return np.array([0]), v[0]
        r = np.linalg.norm(v)
        if r == 0:
            return np.zeros(n-1), 0.0
        angles = np.zeros(n-1)
        cum = 0.0
        for i in range(n-1):
            cum += v[i]**2
            angles[i] = np.arctan2(np.sqrt(r**2 - cum), v[i])
        return angles, r

    def inverse_polar_transform(self, angles, r):
        n = len(angles)+1
        v = np.zeros(n)
        for i in range(n-1):
            prod = r
            for j in range(i): prod *= np.sin(angles[j])
            v[i] = prod * np.cos(angles[i])
        prod = r
        for j in range(n-1): prod *= np.sin(angles[j])
        v[-1] = prod
        return v

    def sample_deflection_angle(self, vec):
        r = np.linalg.norm(vec)
        lam = 1.0 / max(r, 1e-6)
        u = np.random.uniform(-0.5, 0.5)
        sign = np.sign(u)
        abs_theta = -np.log(1 - 2*abs(u)) / lam if abs(u) < 0.5 else 0.0
        return sign * abs_theta

    def predict(self, k2, k1):
        if len(k2) != len(k1):
            return k1
        out = np.zeros_like(k1)
        for i in range(len(k1)):
            vec = k1[i] - k2[i]
            ang, r = self.polar_transform(vec)
            da = self.sample_deflection_angle(vec)
            new_ang = ang + da
            out[i] = k1[i] + self.inverse_polar_transform(new_ang, r)
        return np.clip(out, 0, 1)


# ============================================================================
# Imbalanced transfer learning
# ============================================================================
class ImbalancedTransferLearner:
    def __init__(self, part_num, max_iter=6):
        self.part_num = part_num
        self.max_iter = max_iter
        self.models = []
        self.betas = []

    def _init_weights(self, n_src, n_tgt, src_y):
        w = np.zeros(n_src + n_tgt)
        n_k = np.sum(src_y == 1)
        n_nk = n_src - n_k
        for i in range(n_src):
            w[i] = 1.0/self.part_num if src_y[i]==1 else 1.0/max(n_nk,1)
        w[n_src:] = 1.0/max(n_tgt,1)
        return w / np.sum(w)

    def _train(self, X, y, w):
        valid = ~np.isnan(X).any(axis=1)
        X, y, w = X[valid], y[valid], w[valid]
        if len(X) == 0:
            raise ValueError("no valid sample")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        svm.fit(X, y, sample_weight=w)
        tree.fit(X, y, sample_weight=w)
        return svm, tree

    def fit(self, src_X, src_y, tgt_X, tgt_y=None):
        n_src, n_tgt = len(src_X), len(tgt_X)
        if tgt_y is None:
            tgt_y = np.ones(n_tgt)
        X = np.vstack([src_X, tgt_X])
        y = np.hstack([src_y, tgt_y])
        src_idx = np.arange(n_src)
        tgt_idx = np.arange(n_src, n_src+n_tgt)
        w = self._init_weights(n_src, n_tgt, src_y)
        self.models.clear()
        self.betas.clear()
        for _ in range(self.max_iter):
            svm, tree = self._train(X, y, w)
            py_svm = svm.predict(X)
            py_tree = tree.predict(X)
            py = ((py_svm + py_tree) >= 1).astype(int)
            err = np.sum(w[tgt_idx] * (py[tgt_idx] != y[tgt_idx])) / max(np.sum(w[tgt_idx]), 1e-6)
            err = np.clip(err, 0.001, 0.499)
            beta = err / (1 - err)
            sigma = 1.0
            for i in src_idx:
                e = py[i] != y[i]
                w[i] *= (sigma if y[i]==1 else 1) * (beta**e)
            for i in tgt_idx:
                e = py[i] != y[i]
                w[i] *= beta**(-e)
            w /= np.sum(w)
            self.models.append((svm, tree))
            self.betas.append(beta)
        return self

    def predict_scores(self, X):
        if not self.models:
            return np.zeros(len(X))
        score = np.zeros(len(X))
        n = len(self.models)
        start = max(0, n//2)
        for i in range(start, n):
            svm, tree = self.models[i]
            b = self.betas[i]
            w = np.log(1.0/max(b, 1e-6))
            p1 = svm.predict_proba(X)[:, 1]
            p2 = tree.predict_proba(X)[:, 1]
            score += w * (p1 + p2)/2
        return score


# ============================================================================
# KT-DMOEA main class (batch version)
# ============================================================================
class KT_DMOEAScheduler:
    def __init__(self, problem, pop_size=50, max_gen=100, part_num=10, 
                 batch_size=10, transfer_max_iter=10):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.part_num = part_num
        self.batch_size = batch_size
        self.transfer_max_iter = transfer_max_iter
        
        self.n_var = problem.num_jobs * problem.num_machines
        self.knee_extractor = KneePointExtractor(part_num)
        self.tpm = TPMPredictor(self.n_var)
        
        # Batch-related
        self.knee_array = []      # Per-batch knee history
        self.archive = []         # Global archive
        self.batch_archives = []  # Per-batch non-dominated sets
        
        # Global schedule
        self.global_schedule = {
            'machine_schedules': [[] for _ in range(problem.num_machines)],
            'job_completion_times': [0] * problem.num_jobs
        }
        self.time_offset = 0

    # ===================== Individual operations =====================
    def _create_individual(self, num_jobs):
        individual = []
        for job_id in range(num_jobs):
            individual.extend([job_id] * self.problem.num_machines)
        random.shuffle(individual)
        return individual

    def _repair_individual(self, individual, num_jobs):
        from collections import Counter
        expect = self.problem.num_machines
        cnt = Counter(individual)
        miss, more = [], []
        for j in range(num_jobs):
            c = cnt.get(j, 0)
            if c < expect:
                miss += [j] * (expect - c)
            elif c > expect:
                more += [j] * (c - expect)
        out = individual.copy()
        m = miss.copy()
        for i in range(len(out)):
            if out[i] in more and m:
                more.remove(out[i])
                out[i] = m.pop(0)
        return out

    def _evaluate_individual(self, individual, num_jobs):
        individual = self._repair_individual(individual, num_jobs)
        sch = decode_schedule(self, individual)
        if not sch or not sch.get("valid"):
            return Solution(individual, (float("inf"),)*3, {})
        try:
            obj = evaluate_objectives(self, sch)
        except:
            obj = (float("inf"),)*3
        return Solution(individual, obj, sch)

    def _evaluate_population(self, pop, num_jobs):
        return [self._evaluate_individual(x, num_jobs) for x in pop]

    def _generate_random_population(self, num_jobs, n=None):
        if n is None:
            n = self.pop_size
        return [self._create_individual(num_jobs) for _ in range(n)]

    # ===================== Non-dominated sorting =====================
    def _dominates(self, a, b):
        ok = all(x <= y for x, y in zip(a, b))
        better = any(x < y for x, y in zip(a, b))
        return ok and better

    def _fast_non_dominated_sort(self, sols):
        if not sols:
            return [[]]
        n = len(sols)
        dom_count = [0] * n
        dom_set = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(sols[i].objectives, sols[j].objectives):
                    dom_set[i].append(j)
                elif self._dominates(sols[j].objectives, sols[i].objectives):
                    dom_count[i] += 1
        fronts = [[]]
        for i in range(n):
            if dom_count[i] == 0:
                fronts[0].append(sols[i])
        cur = 0
        while cur < len(fronts) and fronts[cur]:
            nxt = []
            for s in fronts[cur]:
                i = sols.index(s)
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt.append(sols[j])
            if nxt:
                fronts.append(nxt)
            cur += 1
        return fronts

    def _crowding_distance(self, front):
        if len(front) <= 2:
            return [float("inf")] * len(front)
        n = len(front)
        dist = np.zeros(n)
        objs = np.array([s.objectives for s in front])
        for m in range(objs.shape[1]):
            idx = np.argsort(objs[:, m])
            dist[idx[0]] = float("inf")
            dist[idx[-1]] = float("inf")
            rng = objs[idx[-1], m] - objs[idx[0], m]
            if rng < 1e-6:
                continue
            for i in range(1, n-1):
                dist[idx[i]] += (objs[idx[i+1], m] - objs[idx[i-1], m]) / rng
        return dist.tolist()

    def _update_archive(self, sols, archive_size):
        valid = [s for s in sols if all(np.isfinite(s.objectives))]
        total = self.archive + valid
        if not total:
            self.archive = []
            return
        fronts = self._fast_non_dominated_sort(total)
        new_arc = []
        for f in fronts:
            if len(new_arc) + len(f) <= archive_size:
                new_arc += f
            else:
                rem = archive_size - len(new_arc)
                if rem <= 0:
                    break
                d = self._crowding_distance(f)
                sidx = np.argsort(d)[::-1]
                new_arc += [f[i] for i in sidx[:rem]]
                break
        self.archive = new_arc

    def _get_pareto_front(self):
        if not self.archive:
            return np.array([])
        fs = self._fast_non_dominated_sort(self.archive)
        if fs:
            return np.array([s.objectives for s in fs[0]])
        return np.array([])

    # ===================== Encoding/decoding =====================
    def _encode_individual(self, individual, num_jobs):
        maxj = num_jobs - 1
        return np.array([j/maxj if maxj else 0.0 for j in individual])

    def _decode_individual(self, encoded, num_jobs):
        enc = np.asarray(encoded).flatten()
        nm = self.problem.num_machines
        pos = np.argsort(enc)
        prio = {idx % num_jobs: i for i, idx in enumerate(pos[:num_jobs])}
        out = []
        for j in range(num_jobs):
            out += [j] * nm
        out.sort(key=lambda x: prio.get(x, num_jobs))
        return out

    def _select_elite(self, n_elite):
        if not self.archive:
            return []
        take = self.archive[:n_elite]
        return [s.individual for s in take]

    # ===================== Batch core methods =====================
    def _create_batch_problem(self, batch_job_indices, time_offset):
        """Create batch problem instance."""
        batch_arrival = []
        batch_due = []
        batch_machine_seq = []
        batch_process_times = []
        
        for global_job_id in batch_job_indices:
            job_data = self.problem.jobs_data[global_job_id]
            batch_arrival.append(job_data['arrival_time'] + time_offset)
            batch_due.append(job_data['due_date'] + time_offset)
            batch_machine_seq.append(job_data['machine_sequence'])
            batch_process_times.append(job_data['processing_times'])
        
        return JobShopProblem(
            num_jobs=len(batch_job_indices),
            num_machines=self.problem.num_machines,
            arrival_time=batch_arrival,
            due_date=batch_due,
            machine_sequence=batch_machine_seq,
            processing_times=batch_process_times
        )

    def _build_source_domain(self, batch_archive):
        """Build source domain from historical batches."""
        if len(self.knee_array) < 2:
            return np.array([]), np.array([])
        
        pos = []
        for arr in self.knee_array[-2:]:  # Last two batches
            if len(arr) > 0:
                pos.extend(arr.tolist())
        
        if not pos:
            return np.array([]), np.array([])
        
        # Negative samples from current archive
        all_enc = [self._encode_individual(s.individual, batch_archive[0].individual[0] if batch_archive else 10) 
                   for s in batch_archive]
        pset = set(tuple(x) for x in pos)
        neg = [x for x in all_enc if tuple(x) not in pset]
        
        n = min(len(pos), len(neg))
        if n < 2:
            return np.array([]), np.array([])
        
        X = np.vstack([pos[:n], neg[:n]])
        y = np.hstack([np.ones(n), np.zeros(n)])
        return X, y

    def _generate_target_X(self, predicted_knee, batch_problem):
        """Generate target domain samples."""
        tx = []
        if len(predicted_knee) > 0:
            tx.extend(predicted_knee.tolist())
        
        # Add random points
        n_random = max(30, self.pop_size)
        for _ in range(n_random):
            random_point = np.random.rand(batch_problem.num_jobs * batch_problem.num_machines)
            tx.append(random_point)
        
        return np.array(tx)

    def _generate_predicted_population(self, src_X, src_y, tgt_X, batch_problem):
        """Key: predict population via transfer learning."""
        if len(src_X) < 10:
            n = min(self.part_num, len(tgt_X))
            return [self._decode_individual(tgt_X[i], batch_problem.num_jobs) for i in range(n)]
        
        try:
            learner = ImbalancedTransferLearner(self.part_num, self.transfer_max_iter)
            learner.fit(src_X, src_y, tgt_X)
            scores = learner.predict_scores(tgt_X)
            top = np.argsort(scores)[-self.part_num:]
            return [self._decode_individual(tgt_X[i], batch_problem.num_jobs) for i in top]
        except:
            n = min(self.part_num, len(tgt_X))
            return [self._decode_individual(tgt_X[i], batch_problem.num_jobs) for i in range(n)]

    def _moead_optimize(self, init_sols, max_iter, num_jobs):
        """Key: MOEA/D optimization."""
        pop = init_sols[:self.pop_size]
        while len(pop) < self.pop_size:
            pop.append(self._evaluate_individual(self._create_individual(num_jobs), num_jobs))
        
        w = np.random.dirichlet([1, 1, 1], self.pop_size)
        T = min(15, max(5, self.pop_size // 10))
        dist = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2)
        nb = np.argsort(dist, axis=1)[:, :T]
        objs = np.array([s.objectives for s in pop])
        z = np.min(objs, axis=0)
        
        def ch(s, w):
            return np.max(w * np.abs(s - z))
        
        g = [ch(s.objectives, w[i]) for i, s in enumerate(pop)]
        
        for gen in range(max_iter):
            for i in range(self.pop_size):
                a, b = np.random.choice(nb[i], 2, replace=False)
                p1 = pop[a].individual
                p2 = pop[b].individual
                c = p1.copy()
                if random.random() < 0.8:
                    x, y = sorted(random.sample(range(len(c)), 2))
                    c[x:y] = p2[x:y]
                if random.random() < 0.1:
                    x, y = random.sample(range(len(c)), 2)
                    c[x], c[y] = c[y], c[x]
                cs = self._evaluate_individual(c, num_jobs)
                z = np.minimum(z, cs.objectives)
                for j in nb[i]:
                    if ch(cs.objectives, w[j]) <= g[j]:
                        pop[j] = cs
                        g[j] = ch(cs.objectives, w[j])
            
            if gen % 5 == 0 and gen > 0:
                rand = self._evaluate_population(
                    self._generate_random_population(num_jobs, 5), num_jobs
                )
                comb = pop + rand
                fs = self._fast_non_dominated_sort(comb)
                newpop = []
                for f in fs:
                    if len(newpop) + len(f) <= self.pop_size:
                        newpop += f
                    else:
                        rem = self.pop_size - len(newpop)
                        d = self._crowding_distance(f)
                        sidx = np.argsort(d)[::-1]
                        newpop += [f[i] for i in sidx[:rem]]
                        break
                pop = newpop
            self._update_archive(pop, self.pop_size)
        
        return pop

    def _optimize_batch(self, batch_problem, batch_idx, is_first_batch):
        """Optimize a single batch."""
        num_jobs = batch_problem.num_jobs
        print(f"  Batch {batch_idx + 1}: {num_jobs} jobs")
        
        # Initialize population
        if is_first_batch:
            pop_rand = self._generate_random_population(num_jobs, self.pop_size)
            sols = self._evaluate_population(pop_rand, num_jobs)
        else:
            # Initialize with transfer learning
            src_X, src_y = self._build_source_domain(self.archive)
            pred_knee = np.array([])
            if len(self.knee_array) >= 2:
                k1 = self.knee_array[-1]
                k2 = self.knee_array[-2]
                if len(k1) == len(k2) and len(k1) > 0:
                    pred_knee = self.tpm.predict(k2, k1)
            tgt_X = self._generate_target_X(pred_knee, batch_problem)
            pred_pop = []
            if len(src_X) > 0 and len(np.unique(src_y)) == 2:
                pred_pop = self._generate_predicted_population(src_X, src_y, tgt_X, batch_problem)
            
            # Population mixing
            n_rand = int(self.pop_size * 0.2)
            n_elite = int(self.pop_size * 0.3)
            n_pred = self.pop_size - n_rand - n_elite
            
            rand_part = self._generate_random_population(num_jobs, n_rand)
            elite_part = self._select_elite(n_elite)
            pred_part = pred_pop[:n_pred] if pred_pop else []
            
            init = rand_part + elite_part + pred_part
            while len(init) < self.pop_size:
                init.append(self._create_individual(num_jobs))
            
            sols = self._evaluate_population(init, num_jobs)
        
        # MOEA/D optimization
        opt = self._moead_optimize(sols, self.max_gen, num_jobs)
        self._update_archive(opt, self.pop_size)
        
        # Extract knee points
        pf = self._get_pareto_front()
        if len(pf) >= 2:
            kp_obj = self.knee_extractor.get_knee_points(pf)
        else:
            kp_obj = pf
        
        kp_dec = []
        for ko in kp_obj:
            for s in self.archive:
                if np.allclose(s.objectives, ko, atol=1e-3):
                    kp_dec.append(self._encode_individual(s.individual, num_jobs))
                    break
        
        self.knee_array.append(np.array(kp_dec) if kp_dec else np.array([]))
        self.batch_archives.append(self.archive.copy())
        
        return self.archive

    def _merge_batch_result(self, batch_archive, batch_job_indices):
        """Merge batch result into global schedule."""
        if not batch_archive:
            return
        
        # Select batch-best (first non-dominated)
        best_sol = batch_archive[0]
        
        # Apply time offset
        if best_sol.schedule:
            for machine_id in range(len(self.global_schedule['machine_schedules'])):
                for op in best_sol.schedule.get('machine_schedules', []):
                    if machine_id < len(best_sol.schedule['machine_schedules']):
                        for op_info in best_sol.schedule['machine_schedules'][machine_id]:
                            global_op = op_info.copy()
                            global_op['job_id'] = batch_job_indices[op_info['job_id']]
                            global_op['start_time'] += self.time_offset
                            global_op['end_time'] += self.time_offset
                            self.global_schedule['machine_schedules'][machine_id].append(global_op)
            
            # Update completion times
            for batch_job_id, global_job_id in enumerate(batch_job_indices):
                if batch_job_id < len(best_sol.schedule.get('job_completion_times', [])):
                    completion = best_sol.schedule['job_completion_times'][batch_job_id]
                    self.global_schedule['job_completion_times'][global_job_id] = completion + self.time_offset
            
            # Update time offset
            if best_sol.schedule.get('job_completion_times'):
                batch_makespan = max(best_sol.schedule['job_completion_times'])
                self.time_offset += batch_makespan

    # ===================== Entry point =====================
    def run(self) -> Dict:
        """Run KT-DMOEA batch optimization."""
        start_time = time.time()
        
        print("=" * 60)
        print("KT-DMOEA batch optimization started")
        print(f"Jobs: {self.problem.num_jobs}, Machines: {self.problem.num_machines}")
        print(f"Population: {self.pop_size}, Max gen: {self.max_gen}")
        print(f"Subspaces: {self.part_num}, Batch size: {self.batch_size}")
        print("=" * 60)
        
        # Batch processing
        all_jobs = list(range(self.problem.num_jobs))
        num_batches = (len(all_jobs) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(all_jobs))
            batch_job_indices = all_jobs[batch_start:batch_end]
            
            print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
            print(f"  Jobs: {batch_start}-{batch_end-1} ({len(batch_job_indices)} total)")
            
            # Create batch problem instance
            batch_problem = self._create_batch_problem(batch_job_indices, self.time_offset)
            
            # Optimize current batch
            is_first_batch = (batch_idx == 0)
            batch_archive = self._optimize_batch(batch_problem, batch_idx, is_first_batch)
            
            # Merge into global schedule
            self._merge_batch_result(batch_archive, batch_job_indices)
            
            # Batch summary
            pf = self._get_pareto_front()
            print(f"  Batch done | Non-dominated: {len(self.archive)} | Knees: {len(self.knee_array[-1]) if self.knee_array else 0}")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("KT-DMOEA optimization complete")
        print(f"Total time: {total_time:.2f} s")
        print(f"Final non-dominated count: {len(self.archive)}")
        print("=" * 60)
        
        return {
            'pareto_front': [s.objectives for s in self.archive],
            'total_elapsed_time': total_time,
            'archive': self.archive,
            'batch_archives': self.batch_archives,
            'global_schedule': self.global_schedule
        }


# ============================================================================
# Test code
# ============================================================================
if __name__ == "__main__":
    problem = JobShopProblem(num_jobs=50, num_machines=5)
    scheduler = KT_DMOEAScheduler(
        problem=problem,
        pop_size=50,
        max_gen=100,
        part_num=10,
        batch_size=10
    )
    result = scheduler.run()
    
    print(f"\nDone. Valid solution count: {len(result['pareto_front'])}")
    if len(result['pareto_front']) > 0:
        print("Pareto front examples (first 5):")
        for i, obj in enumerate(result['pareto_front'][:5]):
            print(f"  {i+1}: {obj}")