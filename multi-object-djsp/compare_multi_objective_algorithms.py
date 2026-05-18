# main_training_SI.py
import simpy
import sys
sys.path
import random
import numpy as np
import torch
from common.experiment_scene import get_all_scenarios
from common.shared_modules import ShopFloor
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
from algorithm.SI import Tr_DMOEA, FCP, DMOA_MHKT,KT_DMOEA
import importlib
import time 

def run_algorithm(algorithm_module, job_kwargs, run_id_base, mo_manager):
    # Create problem instance
    start_time = time.time()
    job = algorithm_module.JobShopProblem(**job_kwargs)
     # Create optimizer and run
    module_name = algorithm_module.__name__.split('.')[-1]  # Use the last segment as the module name
    optimizer_class = getattr(algorithm_module, f"{module_name}Scheduler")
    #optimizer = optimizer_class(job)
    optimizer = optimizer_class(job,pop_size=100, max_gen=150)
    results = optimizer.run()
    best_solutions = results['pareto_front']
    # Store results
    algorithm_name = module_name
    for i, solution in enumerate(best_solutions):
        run_id = f"{run_id_base}"
        job_record = {'objectives': tuple(solution)}
        mo_manager.add_experiment_data(algorithm_name, run_id, [job_record])
   
    return (time.time()-start_time)/(job_kwargs['num_jobs'] * job_kwargs['num_machines'])
# Training stage remains unchanged
out_analysis = 1
run_num = 10
job_num = 50
drl_preference_runs = 50
baseline = 'TBHL'
# benchmark = ['KT_DMOEA']
benchmark = ['KT_DMOEA','DMOA_MHKT','FCP','Tr_DMOEA']
all_scenarios = get_all_scenarios() 
brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}") 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']              
    print(f'\n=== Multi-objective comparison: scenario={scenario_id} ===')
    for cyc in range(run_num): 
        algorithm_times = {}
        mo_manager = MultiObjectiveManager()        
        print(f'*** Run {cyc+1}/{run_num} ***')
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        drl_start_time = time.time()
        for pref_run  in range(drl_preference_runs): 
            seed = np.random.randint(20000*(1+random.random())) 
            address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt" 
            print(f'DRL preference test {pref_run+1}/{drl_preference_runs}:')
            perturbed_pref =  perturbed_preferences[pref_run]            
            env = simpy.Environment()
            spf = ShopFloor(env, job_num,scenario['parameters'],brain_machine, 
                 seed=seed,preference_vector=perturbed_pref, address=address)
            # spf.job_creator.arrival_interval = [0] * job_num  # Static multi-objective
            spf.simulation(bit=1)    
            run_id = f'{cyc}_{pref_run}'
            mo_manager.add_experiment_data("DRL", run_id, spf.job_objectives_records) 

        # ===== DRL diagnostics (after DRL loop) =====
        if "DRL" in mo_manager.all_experiment_data:
            vals = []
            for rid, recs in mo_manager.all_experiment_data["DRL"].items():
                for r in recs:
                    if 'objectives' in r:
                        vals.append(np.array(r['objectives'], dtype=float))
            if len(vals) > 0:
                arr = np.array(vals)
                print(f"[Diag][DRL] points={len(arr)}")
                print(f"[Diag][DRL] f1 min/mean/max: {arr[:,0].min():.3f}/{arr[:,0].mean():.3f}/{arr[:,0].max():.3f}")
                print(f"[Diag][DRL] f2 min/mean/max: {arr[:,1].min():.3f}/{arr[:,1].mean():.3f}/{arr[:,1].max():.3f}")
                print(f"[Diag][DRL] f3 min/mean/max: {arr[:,2].min():.3f}/{arr[:,2].mean():.3f}/{arr[:,2].max():.3f}")
            else:
                print("[Diag][DRL] No objective data yet")

        algorithm_times['DRL'] = (time.time() - drl_start_time)/(job_num*scenario['parameters']['machine_count'])



        job_kwargs = {'num_jobs': job_num,
                      'num_machines': scenario['parameters']['machine_count'],
            'arrival_time': spf.job_creator.arrival_list,
            'due_date': spf.job_creator.due_list,
            'machine_sequence': spf.job_creator.sequence_list,
            'processing_times': spf.job_creator.pt_list}        
        
          
        for rule in benchmark:    
            print(f"\n=== Scenario {scenario_id}, {rule} multi-objective optimization ===")
            run_id_base = f'{scenario_id}_{cyc}'
            # Key: store each algorithm run under its own label
            algorithm_times[f'{rule}'] = run_algorithm(globals()[rule], job_kwargs, run_id_base, mo_manager)     

            # Diagnostics: objective ranges
            if rule in mo_manager.all_experiment_data:
                vals = []
                for rid, recs in mo_manager.all_experiment_data[rule].items():
                    for r in recs:
                        if 'objectives' in r:
                            vals.append(np.array(r['objectives'], dtype=float))
                if len(vals) > 0:
                    arr = np.array(vals)
                    print(f"[Diag][{rule}] f1 min/mean/max: {arr[:,0].min():.3f}/{arr[:,0].mean():.3f}/{arr[:,0].max():.3f}")
                    print(f"[Diag][{rule}] f2 min/mean/max: {arr[:,1].min():.3f}/{arr[:,1].mean():.3f}/{arr[:,1].max():.3f}")
                    print(f"[Diag][{rule}] f3 min/mean/max: {arr[:,2].min():.3f}/{arr[:,2].mean():.3f}/{arr[:,2].max():.3f}")
                else:
                    print(f"[Diag][{rule}] No objective data yet")
        
        
        print("\n=== Multi-objective performance analysis ===")
        hypervolumes, coverages = mo_manager.get_metrics()
        for rule_name in hypervolumes.keys():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume (HV): {hypervolumes[rule_name]:.6f}")
            print(f"  Coverage: {coverages[rule_name]:.2f}%")
        out_list = ['DRL'] + benchmark

        mo_manager.save_to_excel(scenario_id, cyc, out_list, hypervolumes, coverages, algorithm_times, cpath='DMO')

if out_analysis:    
    data_analysis_report(sub_path= 'DMO')


       