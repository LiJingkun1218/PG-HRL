# cfunctions.py - shared scheduling utilities
import numpy as np
import torch
import agent.sequencing as sequencing
from typing import List, Tuple, Dict
import random

import copy

avg_rwd_rule = [0]
avg_rwd_alpha = [0]

def add_job(self, pt, due):
    """Add job info to the machine lists."""
    self.pt_list.append(pt)  # Per-operation processing times
    self.due_list.append(due)  # Job due date
    self.current_pt.append(pt[self.m_idx] if len(pt) > self.m_idx else 0.0)


def remove_job(self, index):
    """Remove job info."""
    del self.pt_list[index]  # Remove processing times
    del self.due_list[index]  # Remove due date
    self.current_pt = [x[self.m_idx] for x in self.pt_list]


def before_operation(self):
    """Compute temporary features for decision/reward before sequencing."""
    self.waiting_jobs = len(self.queue)  # Queue length
    # Time-related features
    time_till_due = [due - self.env.now for due in self.due_list]
    # Convert to Python lists
    current_pt_vals = [float(pt) for pt in self.current_pt]
    remaining_job_pt_vals = [float(rpt) for rpt in self.remaining_job_pt]        
    self.remaining_no_op = [len(x) for x in self.remaining_pt_list]
    # Other features
    self.before_op_remaining_pt = [rpt + cpt for rpt, cpt in zip(remaining_job_pt_vals, current_pt_vals)]
    self.before_op_slack = []
    for i, (ttd, rpt) in enumerate(zip(time_till_due, self.before_op_remaining_pt)):
        total_slack = ttd - rpt
        if i < len(self.remaining_no_op) and self.remaining_no_op[i] > 0:
            per_operation_slack = total_slack / (self.remaining_no_op[i] + 1)  # +1 includes current op
            self.before_op_slack.append(per_operation_slack)
        else:  # No remaining-op info
            self.before_op_slack.append(total_slack)
    self.pt_chosen = current_pt_vals[self.position]
    
    winq_vals = list(map(float, self.winq))
    self.before_op_winq_chosen = (winq_vals[self.position] if self.position < len(winq_vals) else 0.0 )

    self.before_op_winq_loser = (winq_vals[:self.position] + winq_vals[self.position + 1:] if winq_vals and self.position < len(winq_vals) 
        else [] )


def after_operation(self, timebit):
    """Handle job transfer/completion after processing; update stats and experience."""
    if len(self.sequence_list[self.position]):  # Has remaining operations
        remaining_ptl = self.remaining_pt_list.pop(self.position)  # Remaining op times
        remaining_ptl.pop(0)  # Remove current op time
        next_ma = self.sequence_list[self.position][0]  # Next machine index
        transferred_pt = self.pt_list[self.position]  # Cache pt
        transferred_due = self.due_list[self.position]  # Cache due
        # Key: remove job via helper to keep current_pt in sync
        remove_job(self, self.position)
        self.m_list[next_ma].queue.append(self.queue.pop(self.position))  # Move queue id
        popped_sequence = self.sequence_list.pop(self.position)  # Operation list
        popped_sequence.pop(0)  # Drop first op
        self.m_list[next_ma].sequence_list.append(popped_sequence)
        self.m_list[next_ma].remaining_pt_list.append(remaining_ptl)
        add_job(self.m_list[next_ma], pt=transferred_pt, due=transferred_due)
        try:
            self.m_list[next_ma].sufficient_stock.succeed()  # Notify next machine
        except:
            pass
    else:
        del self.queue[self.position]  # Remove queue id
        del self.sequence_list[self.position]  # Remove op list
        del self.remaining_pt_list[self.position]  # Remove remaining op times
        # Key: remove via helper to keep current_pt in sync
        remove_job(self, self.position)
      
        self.job_creator.in_system_job_no -= 1
        if hasattr(self,'sqc_brain') and self.sqc_brain.train == True:           
            multi_obj_manager = self.job_creator.sqc_brain.multi_obj_manager
            solution = multi_obj_manager.evaluate_solution(self.job_creator, self.job_idx)
            if solution is not None:
                archive_updated = multi_obj_manager.update_archive(solution)            
        
    state_update_all(self)  # Update all states
    complete_experience(self, timebit)   


def state_update_all(self):
    """Update all machine state features for scheduling/reward."""
    # Update current processing times
    self.current_pt = [x[self.m_idx] for x in self.pt_list]
    env_now = float(self.env.now)
    self.cumulative_pt = sum(self.current_pt)   
    self.available_time = env_now + self.cumulative_pt
    self.remaining_job_pt = [sum(x) for x in self.remaining_pt_list]
    self.remaining_no_op = [len(x) for x in self.remaining_pt_list]
    self.next_pt = [float(x[0]) if x else 0.0 for x in self.remaining_pt_list]
    self.completion_rate = [max(0.0, (self.no_ops - len(x) - 1) / self.no_ops) for x in self.remaining_pt_list ]
    self.time_till_due = [float(due) - env_now for due in self.due_list]
    self.slack = [(ttd - cp - rjp) / (rno + 1) if (rno + 1) != 0 else 0
        for ttd, cp, rjp, rno in zip(self.time_till_due, self.current_pt, self.remaining_job_pt, self.remaining_no_op)]
    self.average_workcontent = self.cumulative_pt
    self.average_waiting = max(0, self.available_time - env_now)
    # Compute WINQ and AVLM
    self.winq, self.avlm = zip(*[(float(self.m_list[seq[0]].average_workcontent) if seq else 0.0,
        float(self.m_list[seq[0]].average_waiting) if seq else 0.0)
    for seq in self.sequence_list]) if self.sequence_list else ([], [])


def sequencing_data_generation(self):
    """Generate machine features for sequencing decisions."""
    self.sequencing_data = [
    np.array(self.current_pt),
     np.array(self.remaining_job_pt),
     np.array(self.due_list),
    float(self.env.now),
    np.mean(self.completion_rate) if len(self.completion_rate) > 1 else self.completion_rate,
     np.array(self.time_till_due),
     np.array(self.slack),
     np.array(self.winq),
     np.array(self.avlm),
     np.array(self.next_pt),
    [int(item) for item in self.remaining_no_op],
     np.array(self.queue),
     self.m_idx
]
    
    return self.sequencing_data


def update_global_info_progression(self):
    """Update global progression info for scheduling/reward."""
    realized = [max(0, min(float(ttd), 1)) for ttd in self.time_till_due]
    exp = [max(0, min(float(s), 1)) for s in self.slack]

    # Update machine-specific lists
    self.job_creator.comp_rate_list[self.m_idx] = self.completion_rate
    self.job_creator.realized_tard_list[self.m_idx] = realized
    self.job_creator.exp_tard_list[self.m_idx] = exp
    self.job_creator.available_time_list[self.m_idx] = self.available_time

    # Overall completion rate
    all_comp_rates = [rate for comp_list in self.job_creator.comp_rate_list for rate in comp_list]
    self.job_creator.comp_rate = sum(all_comp_rates) / len(all_comp_rates) if all_comp_rates else 0.0

    # Overall realized tardiness rate
    all_realized = [rate for realized_list in self.job_creator.realized_tard_list for rate in realized_list]
    self.job_creator.realized_tard_rate = 1 - (sum(all_realized) / len(all_realized)) if all_realized else 0.0

    # Overall expected tardiness rate
    all_exp = [rate for exp_list in self.job_creator.exp_tard_list for rate in exp_list]
    self.job_creator.exp_tard_rate = 1 - (sum(all_exp) / len(all_exp)) if all_exp else 0.0

    # Update arriving job slack
    if self.slack and 0 <= self.position < len(self.slack):
        self.job_creator.arriving_job_slack_list[self.m_idx] = float(self.slack[self.position])
    else:
        self.job_creator.arriving_job_slack_list[self.m_idx] = 0.0


def state_multi_channel(self, sqc_data):
    """Build multi-channel state representation (tensor)."""
    # -------------------------- 1. Input data --------------------------
    current_pt, completion_rate, time_till_due, slack = sqc_data[0], sqc_data[4], sqc_data[5], sqc_data[6]
    current_queue, current_m_idx, remaining_no_op = sqc_data[-2], sqc_data[-1], sqc_data[10]

    current_m_idx_val = int(current_m_idx)
    local_job_no = len(current_queue)

    # -------------------------- 2. System-level global features (18D) --------------------------
    # (1) Base info
    local_realized_tard_rate = sum(1 for x in time_till_due if x < 0) / local_job_no if local_job_no > 0 else 0.0
    local_comp_rate = np.mean(completion_rate)
    # (2) System-level global info
    all_m_cumulative_pt = np.array([m.cumulative_run_time for m in self.m_list], dtype=np.float32)

    if all_m_cumulative_pt.size > 0:
        bottleneck_m_idx_val = np.argmax(all_m_cumulative_pt)
        total_pt = all_m_cumulative_pt.sum()
        bottleneck_load = all_m_cumulative_pt[bottleneck_m_idx_val] / total_pt if total_pt != 0 else 0.0
        is_current_m_bottleneck = 1 if current_m_idx_val == bottleneck_m_idx_val else 0
    else:
        bottleneck_load, is_current_m_bottleneck = 0.0, 0

    global_tard_job_count = sum((sum(1 for x in m.time_till_due if x < 0) if isinstance(m.time_till_due, list) else (m.time_till_due < 0).sum()) for m in self.m_list)
    global_tard_rate = global_tard_job_count / self.job_creator.in_system_job_no if self.job_creator.in_system_job_no != 0 else 0.0
    global_load_std = all_m_cumulative_pt.std() if all_m_cumulative_pt.size > 1 else 0.0

    # (3) Processing time info
    if len(current_pt) > 0:
        local_pt_mean, local_pt_min, local_pt_max = sum(current_pt)/len(current_pt), min(current_pt), max(current_pt)
        local_pt_std = np.std(current_pt)
        local_pt_CV = local_pt_std / local_pt_mean if local_pt_mean != 0 else 0.0
    else:
        local_pt_mean = local_pt_min = local_pt_max = local_pt_CV = 0.0

    # (4) Due date and slack info
    ttd_mean, ttd_min, ttd_std = np.mean(time_till_due),np.min(time_till_due),np.std(time_till_due) if len(time_till_due) > 0 else (0,0 ,0 )  
    slack_mean, slack_min,slack_std = np.mean(slack),np.min(slack),np.std(slack) if len(slack) > 0 else (0,0 ,0 )

    # (5) Heterogeneity info
    ttd_CV = (ttd_std / ttd_mean if len(time_till_due) > 0 and ttd_mean != 0 else 0.0)
    ttd_CV = np.clip(ttd_CV, -2.0, 2.0)

    slack_CV = (slack_std / slack_mean if len(slack) > 0 and slack_mean != 0 else 0.0)
    slack_CV = np.clip(slack_CV, -2.0, 2.0)
    
    # Key: concatenate non-position features (18D)
    base_info = [float(self.job_creator.in_system_job_no), float(local_job_no), float(local_comp_rate), float(local_realized_tard_rate)]
    global_info = [float(bottleneck_load), float(is_current_m_bottleneck), float(global_tard_rate), float(global_load_std)]
    pt_info = [float(local_pt_mean), float(local_pt_min), float(local_pt_max)]
    ttd_slack_info = [float(ttd_mean), float(ttd_min), float(slack_mean), float(slack_min)]
    heterogeneity_info = [float(local_pt_CV), float(ttd_CV), float(slack_CV)]
    
    # Create tensor
    non_pos_features = torch.tensor(
        base_info + global_info + pt_info + ttd_slack_info + heterogeneity_info,
        dtype=torch.float32
    )
    #non_pos_features = torch.tensor([0]*18, dtype=torch.float32 )
    # Handle NaN and inf
    s_t = torch.nan_to_num(non_pos_features, nan=0.0, posinf=1, neginf=-1)
    
    # Ensure gradient
    s_t = s_t.requires_grad_(True)
    
    return s_t


def get_reward(self):
    # Convert slack list
    slack_list = [float(s) for s in self.before_op_slack]
    avg_slack = np.mean(slack_list) if slack_list else 0.0
    std_slack = np.std(slack_list) if len(slack_list) > 1 else 1.0
    # Convert other lists
    winq_list = [float(w) for w in self.before_op_winq_loser]
    slack_vals = [float(s) for s in self.before_op_slack]
    # Critical levels
    slack_scale = max(10, std_slack * 0.5)
    critical_level = [1 - s / (abs(s) + 50) for s in slack_vals]
    # Selected vs unselected critical levels
    critical_level_chosen = critical_level[self.position] if self.position < len(critical_level) else 0.0
    critical_level_loser = [cl for i, cl in enumerate(critical_level) if i != self.position]
    # Slack reward
    current_pt_vals = [float(cp) for cp in self.current_pt]
    avg_current_pt = np.mean(current_pt_vals[:max(1, len(current_pt_vals) - 1)])
    earned_slack_chosen = avg_current_pt * critical_level_chosen
    consumed_slack_loser = self.pt_chosen * np.mean(critical_level_loser) if critical_level_loser else 0
    rwd_slack = earned_slack_chosen - consumed_slack_loser
    # WINQ reward
    winq_chosen = float(self.before_op_winq_chosen)
    winq_weight = 0.1 + 0.3 * (1 - min(1, std_slack / 30))
    avg_winq = np.mean(winq_list) if winq_list else 0.0
    rwd_winq = (avg_winq - winq_chosen) * 0.2
    # Delay reward (placeholder)
    
    # Rule reward
    rwd_rule = np.clip(rwd_slack + rwd_winq, -1, 1)
    # Handle NaN and update global averages
    def update_avg_rwd(avg_list, rwd_value, max_len):
        if np.isnan(rwd_value):
            return np.mean(avg_list) if avg_list else 0.0
        avg_list.append(rwd_value)
        if len(avg_list) > max_len:
            avg_list.pop(0)
        return np.mean(avg_list)

    rwd_rule = update_avg_rwd(avg_rwd_rule, rwd_rule, self.m_no)
    

    return torch.tensor(rwd_rule, dtype=torch.float32).squeeze()
 
 
def complete_experience(self, timebit):
    """Complete experience storage when a job finishes (SAC)."""
    try:
        # 1. Retrieve incomplete experience
        incomplete_exp = self.job_creator.incomplete_rep_memo[self.m_idx].pop(timebit)
        s_t, a_rule,reward, preference,job_weights = incomplete_exp 
        local_data = sequencing_data_generation(self)  # Processed features
        s_next_t = state_multi_channel(self, local_data)  # Next state
        job_weights_next = self.sqc_brain.multi_obj_manager.calculate_importance(self.job_creator,self.queue)
        #rt_rule = get_reward(self)
        # rt_rule = rt_rule.squeeze()
        complete_exp = [s_t, a_rule, reward, s_next_t, preference,job_weights,job_weights_next]        
        self.sqc_brain.trajectory_buffer.append(complete_exp) 
        if len(self.sqc_brain.trajectory_buffer) > self.sqc_brain.trajectory_buffer_size:
            self.sqc_brain.trajectory_buffer = self.sqc_brain.trajectory_buffer[-self.sqc_brain.trajectory_buffer_size:]
    except (KeyError, Exception):
        pass


def build_experience(self, timebit, m_idx, s_t, a_rule,reward, preference, job_weights):
    """Ensure consistent tensor shapes."""

    self.job_creator.incomplete_rep_memo[m_idx][timebit] = [s_t, a_rule, reward, preference, job_weights]


def calculate_load_balance(self):   
    if self.cur_ops <= 1:
        return 1
    else:
        idle_time_list = np.array(self.job_creator.idle_time[self.job_idx][1:self.cur_ops+1])    
        if np.mean(idle_time_list) !=0:
            return np.std(idle_time_list) #+ np.mean(idle_time_list)
        else:
            return np.std(idle_time_list) #+ np.mean(idle_time_list)
        
   
def decode_schedule(self, individual: List[int]) -> Dict:
    """Key: decode chromosome into schedule (with dynamic delay)."""
    
    individual_copy = individual.copy()
    
    # Compute actual arrival times
    actual_arrival_times = []
    for i in range(self.problem.num_jobs):
        job_data = self.problem.jobs_data[i]
        nominal_arrival = job_data['arrival_time']
        due_date = job_data['due_date']
        total_processing = job_data['total_processing_time']
        buffer_time = max(0, due_date - nominal_arrival - total_processing)
        random_factor = random.random()
        actual_arrival = nominal_arrival + buffer_time * random_factor
        actual_arrival_times.append(actual_arrival)
    
    # Validate chromosome
    for i, job_id in enumerate(individual_copy):
        if job_id is None:
            print(f"Warning: None value found at position {i} in individual")
            return 
        
        if not isinstance(job_id, int):
            print(f"Warning: Non-integer job_id {job_id} found at position {i}")
            return 
            
        if job_id < 0 or job_id >= len(self.problem.jobs_data):
            print(f"Warning: job_id {job_id} out of range [0, {len(self.problem.jobs_data)-1}]")
            return 
    
    num_machines = self.problem.num_machines
    unique_jobs = set(individual_copy)
    for job_id in unique_jobs:
        count = individual_copy.count(job_id)
        if count != num_machines:
            print(f"Warning: job_id {job_id} appears {count} times, expected {num_machines} times")
            return 
    
    # Initialize data structures
    machine_schedules = [[] for _ in range(self.problem.num_machines)]
    
    # Initialize per-job waiting history
    # Format: {job_id: [wait_1, wait_2, ...]} between consecutive ops
    job_waiting_history = [[] for _ in range(self.problem.num_jobs)]
    
    # Record op start/end per job
    job_operation_times = [{} for _ in range(self.problem.num_jobs)]  # {op_index: {'start': x, 'end': y}}
    
    # Track per-machine slack
    machine_slack_history = [[] for _ in range(self.problem.num_machines)]
    
    # Initialize job progress
    job_progress = [{
        'current_op': 0,
        'completion_time': actual_arrival_times[i],
        'last_operation_end_time': actual_arrival_times[i]  # Last op end time
    } for i in range(self.problem.num_jobs)]
    
    job_operations = [[] for _ in range(self.problem.num_jobs)]
    
    # Scheduling loop
    scheduled_count = 0
    total_operations = len(individual_copy)
    max_retry = total_operations * 2
    retry_count = 0
    
    while scheduled_count < total_operations and retry_count < max_retry:
        retry_count += 1
        scheduled_in_this_round = False
        
        for pos in range(len(individual_copy)):
            job_id = individual_copy[pos]
            if job_id == -1:
                continue
                
            job_data = self.problem.jobs_data[job_id]
            op_index = job_progress[job_id]['current_op']
            
            if op_index >= self.problem.num_machines:
                individual_copy[pos] = -1
                scheduled_count += 1
                scheduled_in_this_round = True
                continue
                
            machine_id = job_data['machine_sequence'][op_index]
            processing_time = job_data['processing_times'][op_index]
            
            job_ready_time = job_progress[job_id]['completion_time']
            
            if machine_schedules[machine_id]:
                machine_ready_time = machine_schedules[machine_id][-1]['end_time']
            else:
                machine_ready_time = 0
            
            # ============ Allowed delay computation ============
            allowed_delay = 0
            current_time = max(machine_ready_time, job_ready_time)
            
            # Theoretical delay based on due date
            due_date = job_data['due_date']
            remaining_ops = self.problem.num_machines - op_index - 1  # Remaining ops count
            
            # Remaining processing time
            if op_index < self.problem.num_machines - 1:
                remaining_processing = sum(job_data['processing_times'][op_index + 1:])
            else:
                remaining_processing = 0  # Last operation
            
            allowed_delay = due_date - current_time - processing_time - remaining_processing
            
            # Actual delay
            actual_delay = 0
            
            if op_index == 0:  # First operation
                # Evenly allocate delay
                if allowed_delay > 0:
                    actual_delay = allowed_delay / num_machines
                else:
                    actual_delay = 0
                    
                # Record first-op delay
                if 'idle_time' not in locals():
                    idle_time = {}
                idle_time[job_id] = [actual_delay] + [0] * (self.problem.num_machines - 1)
                    
            else:  # Non-first operation
                if allowed_delay > 0:
                    if job_waiting_history[job_id]:  # Has waiting history
                        # Mean waiting time of completed ops
                        mean_waiting_time = sum(job_waiting_history[job_id]) / len(job_waiting_history[job_id])
                        
                        # Elapsed since last op
                        elapsed_time = current_time - job_progress[job_id]['last_operation_end_time']
                        
                        # Similar to calculate_delay
                        actual_delay = max(0, min(mean_waiting_time, allowed_delay) - elapsed_time)
                    else:  # No history, use random proportion
                        actual_delay = allowed_delay * random.random()
                else:
                    actual_delay = 0
            
            # Clamp delay
            actual_delay = max(0, min(actual_delay, max(0, allowed_delay)))
            
            # Actual start time
            start_time = max(machine_ready_time, job_ready_time) + actual_delay
            end_time = start_time + processing_time
            # ============ Delay computation end ============
            
            # Record waiting time correctly
            if op_index > 0:
                # Current start - previous end
                wait_time = start_time - job_progress[job_id]['last_operation_end_time']
                job_waiting_history[job_id].append(wait_time)
            
            # Record op times
            job_operation_times[job_id][op_index] = {
                'start_time': start_time,
                'end_time': end_time,
                'wait_time': wait_time if op_index > 0 else 0
            }
            
            # Record machine slack
            slack_time = max(0, allowed_delay) - actual_delay
            machine_slack_history[machine_id].append(slack_time)
            
            # Operation info
            operation_info = {
                'job_id': job_id,
                'operation_index': op_index,
                'start_time': start_time,
                'end_time': end_time,
                'processing_time': processing_time,
                'allowed_delay': max(0, allowed_delay),
                'actual_delay': actual_delay,
                'wait_time': wait_time if op_index > 0 else 0
            }
            
            machine_schedules[machine_id].append(operation_info)
            
            job_op_info = {
                'machine_id': machine_id,
                'operation_index': op_index,
                'start_time': start_time,
                'end_time': end_time,
                'processing_time': processing_time,
                'actual_arrival_time': actual_arrival_times[job_id],
                'allowed_delay': max(0, allowed_delay),
                'actual_delay': actual_delay,
                'wait_time': wait_time if op_index > 0 else 0
            }
            job_operations[job_id].append(job_op_info)
            
            # Update job progress
            job_progress[job_id]['completion_time'] = end_time
            job_progress[job_id]['current_op'] += 1
            job_progress[job_id]['last_operation_end_time'] = end_time  # Update last op end
            
            individual_copy[pos] = -1
            scheduled_count += 1
            scheduled_in_this_round = True
        
        if not scheduled_in_this_round:
            break
    
    # Sort ops by index per job
    for job_id in range(self.problem.num_jobs):
        job_operations[job_id].sort(key=lambda x: x['operation_index'])
    
    # Normalize waiting history per job
    # Index i corresponds to wait between op i and i+1
    formatted_waiting_history = []
    for job_id in range(self.problem.num_jobs):
        # Ensure length equals num_ops - 1
        expected_len = self.problem.num_machines - 1
        current_len = len(job_waiting_history[job_id])
        
        if current_len < expected_len:
            # Pad missing waits with 0
            job_waiting_history[job_id].extend([0] * (expected_len - current_len))
        elif current_len > expected_len:
            # Truncate extra waits
            job_waiting_history[job_id] = job_waiting_history[job_id][:expected_len]
        
        formatted_waiting_history.append(job_waiting_history[job_id])
    
    result = {
        'machine_schedules': machine_schedules,
        'job_operations': job_operations,
        'job_completion_times': [job_progress[i]['completion_time'] 
                                for i in range(self.problem.num_jobs)],
        'actual_arrival_times': actual_arrival_times,
        'job_waiting_history': formatted_waiting_history,  # Normalized waiting history
        'machine_slack_history': machine_slack_history,
        'job_operation_times': job_operation_times,  # Detailed op time records
        'valid': scheduled_count == total_operations
    }
    
    return result 

    
def evaluate_objectives(self, schedule: Dict) -> Tuple[float, float, float]:
    """Key: evaluate three objectives (using actual arrivals)."""
    job_completion_times = schedule['job_completion_times']
    job_operations = schedule['job_operations']
    actual_arrival_times = schedule['actual_arrival_times']  # Actual arrivals
    
    # Objective 1: total tardiness
    total_tardiness = 0
    for i, completion_time in enumerate(job_completion_times):
        due_date = self.problem.jobs_data[i]['due_date']
        tardiness = abs(completion_time - due_date)
        total_tardiness += tardiness
    
    # Objective 2: total waiting time (actual arrivals)
    total_waiting_time = 0
    for i, job_data in enumerate(self.problem.jobs_data):
        completion_time = job_completion_times[i]
        arrival_time = actual_arrival_times[i]
        total_processing = job_data['total_processing_time']
        waiting_time = (completion_time - arrival_time) - total_processing
        total_waiting_time += max(0, waiting_time)
    
    # Objective 3: CV sum of inter-op waits (actual arrivals)
    total_cv = 0.0
    
    for job_id in range(self.problem.num_jobs):
        job_data = self.problem.jobs_data[job_id]
        # Use actual arrivals
        arrival_time = actual_arrival_times[job_id]
        
        ops = job_operations[job_id]
        
        if not ops:
            job_cv = 0.0
        else:
            inter_op_wait_times = []
            
            # First-op wait (arrival to start)
            first_op = ops[0]
            first_wait = max(0, first_op['start_time'] - arrival_time)
            inter_op_wait_times.append(first_wait)
            
            # Inter-op waits
            for i in range(1, len(ops)):
                prev_op_end = ops[i-1]['end_time']
                curr_op_start = ops[i]['start_time']
                wait_time = max(0, curr_op_start - prev_op_end)
                inter_op_wait_times.append(wait_time)
            
            # Coefficient of variation
            if len(inter_op_wait_times) < 2:
                job_cv = 0.0
            else:
                mean_wait = np.mean(inter_op_wait_times)
                if mean_wait == 0:
                    job_cv = 0.0
                else:
                    std_wait = np.std(inter_op_wait_times)
                    job_cv = std_wait  # Keep original behavior
        
        total_cv += job_cv
    
    return total_tardiness, total_waiting_time, total_cv
 
 