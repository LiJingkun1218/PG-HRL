import sys 
sys.path 
import numpy as np
import torch
from copy import deepcopy 
from tabulate import tabulate 
import agent.sequencing as sequencing
from common.cfunctions import (
    before_operation,calculate_load_balance,add_job,remove_job,
    state_update_all, sequencing_data_generation,
    after_operation,update_global_info_progression    
)

class machine:
    def __init__(self, env, index, *args, **kwargs): 
        self.env = env 
        self.m_idx = index 
        self.queue = [] 
        self.sequence_list = [] 
        self.pt_list = [] # Processing time list for each job on each machine (2D list: rows=jobs, cols=machines)
        self.remaining_pt_list = [] # Remaining processing time list for each job on each machine (2D list; remove first element after each operation)
        self.due_list = [] # Due date for each job (one due date per job)
        self.decision_point = 0        
        self.available_time = 0 
        self.average_workcontent = 0 
        self.delay_records = []
        self.before_op_slack = []
        self.before_op_winq_loser = []        
        self.average_waiting = 0 
        self.cumulative_run_time = 0 
        self.global_exp_tard_rate = 0 
        self.sufficient_stock = self.env.event() # SimPy event: machine has enough jobs to process (handles starvation)
        self.working_event = self.env.event() # SimPy event: machine is in working state (handles breakdowns)
        self.current_pt = [] 
        self.waiting_jobs = 0 
        self.position = 0 
        self.being_time = 0
        self.delay = 0 
        self.job_idx = 0 
        self.total_idle_time = 0
        self.idle_start_time = 0
        self.use_ratio = 1      
        self.ahead_delay_record = np.array([0],dtype=np.float32) # Record of actual job tardiness for this machine (for statistics and reward calculation)
        self.ahead_delay_record_ga = np.array([0],dtype=np.float32) # Tardiness record for GA (separated from DRL to avoid interference)
        self.avg_tardiness = 0 
        if not len(self.queue):
            self.sufficient_stock.succeed() # Immediately trigger sufficient stock event to prevent simulation deadlock
        self.working_event.succeed() # Machine starts in working state (no breakdown by default)
        if 'rule' in kwargs: # If scheduling rule is provided
            order = "self.job_sequencing = sequencing." + kwargs['rule'] # Dynamically bind scheduling rule function
            try:
                exec(order) 
            except:
                raise Exception
        else:
            self.job_sequencing = sequencing.FIFO # Default to FIFO scheduling rule
   
    def initialization(self, machine_list, job_creator):
        self.m_list = machine_list # List of all machines
        self.m_no = len(self.m_list) # Total number of machines
        self.no_ops = len(self.m_list) # Total number of operations per job (assume all jobs have same number)
        self.job_creator = job_creator # Job creator object
        self.cur_ops = 0
        state_update_all(self) 
        update_global_info_progression(self) 
        self.env.process(self.production()) # Start machine production process

    def production(self):
        """Main production loop for the machine - handles first operation delay separately"""
        if not len(self.queue):
            yield self.env.process(self.starvation())
        state_update_all(self)
        while True:
            self.decision_point = self.env.now
            # 1. Select job
            sqc_data = sequencing_data_generation(self)
            self.position, self.delay = self.job_sequencing(sqc_data)
            self.job_idx = self.queue[self.position]
            # 2. Check if first operation
            current_sequence = self.sequence_list[self.position] if self.position < len(self.sequence_list) else []
            self.cur_ops = self.no_ops - len(current_sequence)
            is_first_operation = (self.cur_ops == 1)
            record = self.job_creator.production_record[self.job_idx]
            idle_time = self.job_creator.idle_time[self.job_idx]
            objects = self.job_creator.objects[self.job_idx]
            seq_time = self.job_creator.seq_time[self.job_idx]
            already_delayed = record[4] if record[4] == True else False
            time_bit = False if hasattr(self,'sqc_brain') and self.env.now <= self.sqc_brain.warm_up else True
            # 3. Special handling for first operation delay
            if is_first_operation and self.delay > 0 and not already_delayed and time_bit:
                record[4] = True
                record[0] = self.env.now + self.delay
                # Save job info for later recovery
                saved_sequence = self.sequence_list[self.position].copy() if self.position < len(self.sequence_list) else []
                saved_remaining_pt = self.remaining_pt_list[self.position].copy() if self.position < len(self.remaining_pt_list) else []
                # Remove from queue
                self.queue.pop(self.position)
                self.sequence_list.pop(self.position)
                self.remaining_pt_list.pop(self.position)
                remove_job(self, self.position)
                state_update_all(self)
                # Create independent delay process (non-blocking)
                self.env.process(
                    self._handle_first_op_delay(
                        job_idx=self.job_idx,
                        delay=self.delay,
                        saved_sequence=saved_sequence,
                        saved_remaining_pt=saved_remaining_pt,
                        current_machine=self.m_idx
                    )
                )
                # Continue to next job if queue is empty
                if not len(self.queue):
                    yield self.env.process(self.starvation())
                continue
            # 4. Normal processing (not first operation or no delay)
            before_operation(self)
            pt = self.pt_list[self.position][self.m_idx] if self.position < len(self.pt_list) else 0
            # 5. Wait for processing time (consider delay)
            yield self.env.timeout(pt + self.delay)
            # 6. Update statistics
            self.cumulative_run_time += pt
            record[2] += self.env.now - record[0] - pt
            idle_time[self.cur_ops] = self.env.now - record[0] - pt
            objects[2] = record[2] * self.m_no / self.cur_ops
            record[0] = self.env.now
            seq_time[self.cur_ops] = self.env.now
            record[3] = calculate_load_balance(self)
            objects[3] = record[3]
            record[1] = self.job_creator.due_list[self.job_idx] - self.env.now
            if record[1] >= 0:
                objects[1] = record[1] * self.cur_ops / self.m_no
            else:
                objects[1] = abs(record[1] * (1 + len(current_sequence) / self.m_no))
            # Last operation handling
            if len(current_sequence) == 0:
                record[1] = abs(self.job_creator.due_list[self.job_idx] - self.env.now)
            # 7. Operation transfer
            after_operation(self, self.decision_point)
            update_global_info_progression(self)
            # 8. Check queue
            if not len(self.queue):
                yield self.env.process(self.starvation())

    def _handle_first_op_delay(self, job_idx, delay, saved_sequence, saved_remaining_pt, current_machine):
        """
        Independent process to handle first operation delay.
        Args:
            job_idx: Job index
            delay: Delay time
            saved_sequence: Saved operation sequence
            saved_remaining_pt: Saved remaining processing time
            current_machine: Current machine index
        """
        yield self.env.timeout(delay)
        # Re-add job to current machine queue after delay
        self.queue.append(job_idx)
        self.sequence_list.append(saved_sequence)
        self.remaining_pt_list.append(saved_remaining_pt)
        add_job(self,pt=self.job_creator.pt_list[job_idx],due=self.job_creator.due_list[job_idx])
        state_update_all(self)
        try:
            if not self.sufficient_stock.triggered:
                self.sufficient_stock.succeed()
        except:
            pass
        
        

       
    def update_global_info_after_operation(self): 
        self.job_creator.next_wc_list[self.m_idx] = -1  
   
    def starvation(self): 
        self.sufficient_stock = self.env.event() 
        yield self.sufficient_stock 
        if not self.working_event.triggered: 
            yield self.env.process(self.breakdown()) 
        state_update_all(self) 

    def __deepcopy__(self, memo): 
   
        new_obj = object.__new__(machine) 
        memo[id(self)] = new_obj  #
        
        skip_attrs = {
            'sufficient_stock', 'working_event', 'job_creator','job_sequencing'
        }
        
        for attr_name in self.__dict__: 
            if attr_name in skip_attrs:
                continue
            
            if attr_name == 'env': 
                original_env = getattr(self, attr_name) 
                new_env = type(original_env)()  
                
                if original_env.now > 0:
                    def advance_time(env, target_time):
                        yield env.timeout(target_time - env.now)
                    new_env.process(advance_time(new_env, original_env.now))
                    new_env.run()
                
                setattr(new_obj, attr_name, new_env)
                continue     
            
            attr_value = getattr(self, attr_name)
            setattr(new_obj, attr_name, deepcopy(attr_value, memo))
        
        return new_obj

