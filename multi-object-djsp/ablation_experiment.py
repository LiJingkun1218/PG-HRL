# main_training_rule.py
import simpy
import sys
sys.path
import random
import os
from openpyxl import load_workbook,Workbook
import numpy as np
import torch
from common.experiment_scene import get_all_scenarios
from common.shared_modules import ShopFloor
from common.experimental_analysis import MultiObjectiveManager, data_analysis_report
import importlib


out_analysis = 1     
run_num = 25    
job_num = 50      
drl_preference_runs = 5
baseline = 'PG_HRL'
benchmark = ['PG_HRL','Ablation1','Ablation2','Ablation3','Ablation4']
# Ablation experiment settings:
#Ablation1: removing the start-time delay calculation module
#Ablation2: removing the weight optimization module
#Ablation3: replacing preference guidance with random weights
#Ablation4: replacing the proposed preference generation method with that of PD-MORL

all_scenarios=get_all_scenarios() 
for scenario in all_scenarios:
    scenario_id=scenario['scenario_id']  
    for cyc in range(run_num):
        print(f'*** Scenario:{scenario_id}, Experimental round {cyc+1}/{run_num} ***')  
        mo_manager = MultiObjectiveManager()
        seed = np.random.randint(20000*(1+random.random()))        
        perturbed_preferences = torch.tensor([np.random.dirichlet([1, 1, 1]) for _ in range(drl_preference_runs)],dtype=torch.float32)        
        for rule in benchmark:    
            if rule != baseline:
                address = f"{sys.path[0]}/sequencing_models/{baseline}_{rule}_{scenario_id}.pt"
            else:                   
                address = f"{sys.path[0]}/sequencing_models/{baseline}_{scenario_id}.pt" 
            brain_machine = importlib.import_module(f"algorithm.RL.brain_{baseline}") 
            for pref_run  in range(drl_preference_runs):
                print(f'{rule}test, NO.{cyc}: {pref_run+1}/{drl_preference_runs}:')                       
                env = simpy.Environment()   
                perturbed_pref =  perturbed_preferences[pref_run]                  
                spf = ShopFloor(env, job_num, scenario['parameters'],brain_machine,
                                    seed=seed, ablation=rule,preference_vector=perturbed_pref,address=address)
                spf.simulation()
                run_id = f'{scenario_id}_{cyc}_pref_{pref_run}'
                mo_manager.add_experiment_data(f'{rule}', run_id, spf.job_objectives_records)                
                
        print("\n=== Multi-Objective Performance Analysis ===")
        hypervolumes, coverages = mo_manager.get_metrics()
        for rule_name in hypervolumes.keys():
            print(f"\nRule {rule_name}:")
            print(f"  HV: {hypervolumes[rule_name]:.6f}")
            print(f"  CR: {coverages[rule_name]:.2f}%")

        mo_manager.save_to_excel(scenario_id, cyc, benchmark, hypervolumes, coverages, cpath='Ablation')

if out_analysis:    
    data_analysis_report(sub_path= 'Ablation')
       
            
        