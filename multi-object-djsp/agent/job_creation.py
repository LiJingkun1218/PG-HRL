import numpy as np
import random # Import random module for generating processing time, arrival interval, due date, etc.
from tabulate import tabulate
import matplotlib.pyplot as plt
from common.cfunctions import (add_job)
class creation:
    def __init__ (self, env=None, job_numbers=10000, machine_list=None,  pt_range=[5,15], due_tightness=4, E_utliz=0.8, train = False,GA = None, **kwargs):
        if 'seed' in kwargs: # Set random seed for reproducibility if provided
            np.random.seed(kwargs['seed'])
            random.seed(kwargs['seed'])
         
        self.env = env        
        self.train = train
        self.m_list = machine_list # List of machines
        self.no_machines = len(self.m_list) # Number of machines
        self.production_record = {} # Production info for each job (completion time, tardiness, etc.)
        self.objects = {}  # Objective values for each job
        self.idle_time = {}
        self.seq_time = {}
        self.pt_range = pt_range
        self.avg_pt = (sum(self.pt_range) / len(self.pt_range)) - 0.5 # Average processing time (minus 0.5 to offset integer rounding bias)
        self.span = job_numbers*self.avg_pt
        self.tightness = due_tightness
        self.E_utliz = E_utliz
        self.Idle_ratio = 0
        self.sequence_seed = list(range(self.no_machines))  # Initial seed for job processing order
        self.in_system_job_no = 0 # Number of jobs currently in the system
        self.index_jobs = 0 # Job index, next job's id (starts from 0)
        self.comp_rate_list = [[] for m in self.m_list] # Completion rate list for each machine
        self.comp_rate = 0 # Overall completion rate
        self.realized_tard_list = [[] for m in self.m_list] # Actual tardiness list for each machine
        self.realized_tard_rate = 0 # Overall actual tardiness rate
        self.exp_tard_list = [[] for m in self.m_list] # Expected tardiness list for each machine
        self.exp_tard_rate = 0 # Overall expected tardiness rate
        self.available_time_list = [0 for m in self.m_list] # Available time for each machine
        self.arriving_job_slack_list = [0 for m in self.m_list] # Slack time for arriving jobs on each machine
        self.sequence_list = [] # Job processing sequence
        self.pt_list = [] # Job processing times
        self.due_list = [] # Job due dates
        self.arrival_list = [] # Job arrival times
        self.departure_dict = {} # Job departure times
        self.expected_tardiness_dict = {} # Expected tardiness for each job
        self.beta = self.avg_pt / self.E_utliz # Mean arrival interval, controls arrival rate vs. machine load (target utilization)
        self.total_no = job_numbers
        # Arrival interval generation: Erlang/Gamma distribution
        k = 5  # Number of stages (recommended 3~5, larger k = smoother)
        scale = self.beta / k  # Mean time per stage, total mean = k*scale = beta
        self.arrival_interval = [round(x) for x in np.random.gamma(shape=k, scale=scale, size=self.total_no)]
        # Processing time generation function
        self.ptl_generation = self.ptl_generation_random
        self.sqc_brain = None
        self.initial_job_assignment() # Initial job assignment (hot start)
        self.env.process(self.new_job_arrival()) # Start new job arrival process

    def initial_job_assignment(self):
        """
        Assign one initial job to each machine (hot start), to avoid system-wide idleness at simulation start.
        """
        sqc_seed = list(range(self.no_machines))
        if self.index_jobs < self.total_no:
            for m_idx, m in enumerate(self.m_list):
                random.shuffle(sqc_seed)
                sqc = [m_idx] + [x for x in sqc_seed if x != m_idx]
                self.sequence_list.append(sqc)
                ptl = self.ptl_generation()    
                self.pt_list.append(ptl)
                remaining_ptl = ptl[1:] if len(ptl) > 0 else []
                avg_pt = sum(ptl) / len(ptl) if ptl else 0 
                due = round(avg_pt * self.no_machines * random.uniform(1, self.tightness)) 
                self.due_list.append(due)    
                self.arrival_list.append(int(self.env.now))     
                self.production_record[self.index_jobs] = [0,0,0,0,0] 
                self.objects[self.index_jobs] = [0,0,0,0]
                self.objects[self.index_jobs] = [0,0,0,1]
                self.objects[self.index_jobs][1] = (due - sum(ptl)) / len(ptl)
                self.idle_time[self.index_jobs] = [0] * (self.no_machines+1)
                self.seq_time[self.index_jobs] = [0] * (self.no_machines+1)
                self.in_system_job_no += 1
                m.queue.append(self.index_jobs)
                m.sequence_list.append(sqc[1:])
                m.remaining_pt_list.append(remaining_ptl)
                add_job(m, pt=self.pt_list[self.index_jobs], due=self.due_list[self.index_jobs])
                self.index_jobs += 1
   
    def new_job_arrival(self):
        """
        Continuously generate new jobs while job index is less than total number of jobs.
        """
        while self.index_jobs < self.total_no:
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)
            random.shuffle(self.sequence_seed)
            self.sequence_list.append(self.sequence_seed.copy())
            ptl = self.ptl_generation()
            self.pt_list.append(ptl)
            avg_pt = sum(ptl) / len(ptl) if ptl else 0
            due = round(avg_pt * self.no_machines * random.uniform(1, self.tightness) + self.env.now)
            self.due_list.append(due)
            self.arrival_list.append(int(self.env.now))
            self.in_system_job_no += 1
            first_machine = self.sequence_seed[0]
            self.production_record[self.index_jobs] = [0,0,0,0,0]
            self.objects[self.index_jobs] = [0,0,0,1]
            self.objects[self.index_jobs][1] = (due - sum(ptl)) / len(ptl)
            self.idle_time[self.index_jobs] = [0] * (self.no_machines+1)
            self.seq_time[self.index_jobs] = [0] * (self.no_machines+1)
            self.m_list[first_machine].queue.append(self.index_jobs)
            self.m_list[first_machine].sequence_list.append(self.sequence_list[self.index_jobs][1:])
            self.m_list[first_machine].remaining_pt_list.append(self.pt_list[self.index_jobs][1:])
            add_job(self.m_list[first_machine], pt=self.pt_list[self.index_jobs], due=self.due_list[self.index_jobs])
            try:
                if not self.m_list[first_machine].sufficient_stock.triggered:
                    self.m_list[first_machine].sufficient_stock.succeed()
            except:
                pass
            self.index_jobs += 1
           
    def ptl_generation_random(self):
        """
        Generate a list of processing times for each new job (length = no_machines).
        """
        return [random.randint(self.pt_range[0], self.pt_range[1]-1) for _ in range(self.no_machines)]
        

    def dynamic_seed_change(self, interval):
        """
        Periodically change random seed during simulation to introduce more randomness.
        """
        while self.in_system_job_no >= 1:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
   
      
    def build_sqc_experience_repository(self, m_list):
        """
        Build job sequencing experience repository: for each machine, prepare two containers (incomplete and complete experience).
        """
        self.incomplete_rep_memo = {} # Incomplete job sequence experience (dict by machine_id)
        self.rep_memo = {} # Complete job sequence experience
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}
            self.rep_memo[m.m_idx] = []

    def tardiness_output(self):
        """
        Collect and output tardiness-related statistics for jobs.
        """
        tard_info = []
        for item in self.production_record: 
            record = self.production_record[item]
            
            if len(record) > 2 and record[2] is not None: 
                completion_time = record[2]
                due_date = self.due_list[item] if item < len(self.due_list) else 0
                tardiness = abs(completion_time - due_date)
                
                
                start_time = record[1][0] if record[1] and len(record[1]) > 0 else completion_time
                total_processing_time = sum(self.pt_list[item]) if item < len(self.pt_list) else 1
                flow_time = completion_time - start_time
                flow_time_ratio = flow_time / total_processing_time if total_processing_time > 0 else 1
                
                tard_info.append((completion_time, tardiness, flow_time_ratio))
        
        
        tard_info_sorted = sorted(tard_info, key=lambda x: x[0]) 
        
        
        output_time = [x[0] for x in tard_info_sorted]
        tard = [x[1] for x in tard_info_sorted]
        flow_ratios = [x[2] for x in tard_info_sorted] 
        
        cumulative_tard = [] 
        current_sum = 0 
        for t in tard: 
            current_sum += t 
            cumulative_tard.append(current_sum) 
        
        
        tard_max = max(tard) if tard else 0 
        tard_mean = [cumulative_tard[i] / (i+1) for i in range(len(cumulative_tard))] if cumulative_tard else [] 
        tard_rate = sum(1 for t in tard if t > 0) / len(tard) if tard else 0 
        
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate, flow_ratios
        


