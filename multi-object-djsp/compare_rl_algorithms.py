import simpy
import sys
sys.path #sometimes need this to refresh the path
import importlib
import numpy as np
import random
import torch
import warnings
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
from common.shared_modules import ShopFloor
from common.experiment_scene import get_all_scenarios
warnings.filterwarnings("ignore")

out_analysis = 1  
run_num = 25
job_num = 50
drl_preference_runs = 5
benchmark = ['SAC','TD3','TBHL','DDQN','Rainbow_DQN']  

all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']
    for cyc in range(run_num):  
        print(f'*** Scenario:{scenario_id}, Experimental round {cyc+1}/{run_num} ***')
        mo_manager = MultiObjectiveManager()        
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        for pref_run  in range(drl_preference_runs):             
            seed = np.random.randint(20000*(1+random.random()))            
            for rule in benchmark:  
                brain_machine = importlib.import_module(f"algorithm.RL.brain_{rule}")              
                address = f"{sys.path[0]}/sequencing_models/{rule}_{scenario_id}.pt"
                print(f'{scenario_id}scenario, {rule}test, round {cyc+1}: {pref_run+1}/{drl_preference_runs}:')                      
                env = simpy.Environment()   
                perturbed_pref =  perturbed_preferences[pref_run]                  
                spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                    seed=seed, preference_vector=perturbed_pref,address=address)                      
                spf.simulation()
                run_id = f'{scenario_id}_{cyc}_pref_{pref_run}'
                mo_manager.add_experiment_data(f'{rule}', run_id, spf.job_objectives_records)    
                
        print("\n=== Multi-Objective Performance Analysis ===")
        hypervolumes, coverages = mo_manager.get_metrics()
        for rule_name in hypervolumes.keys():
            print(f"\nRule {rule_name}:")
            print(f"  Hypervolume (HV): {hypervolumes[rule_name]:.6f}")
            print(f"  Coverage: {coverages[rule_name]:.2f}%")

        mo_manager.save_to_excel(scenario_id, cyc, benchmark, hypervolumes, coverages, cpath='Ref_Learning')

if out_analysis:    
    data_analysis_report(sub_path= 'Ref_Learning')
        