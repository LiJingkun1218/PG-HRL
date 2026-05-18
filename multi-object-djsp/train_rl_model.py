# main_training_RL.py
import simpy
import sys
sys.path
from common.experiment_scene import get_all_scenarios
from common.shared_modules import ShopFloor
import importlib
 
benchmark = ['TBHL'] 
#benchmark = ['TBHL','SAC','TD3','DDQN','Rainbow_DQN'] 
#benchmark = ['Ablation1','Ablation2','Ablation3','Ablation4']  #Ablation experiments for different components of TBHL (e.g., without load balance, without delay info, etc.)
job_numbers = 10000
if __name__ == "__main__":   
    all_scenarios=get_all_scenarios() 
    for scenario in all_scenarios:             
        scenario_id=scenario['scenario_id'] 
        for rule in benchmark:
            #DRL Model training
            brain_machine = importlib.import_module(f"algorithm.RL.brain_{rule}") 
            address=f"{sys.path[0]}/sequencing_models/{rule}_{scenario_id}.pt"  
            print(f'\n=== scenario={scenario_id},Algorithm{rule}model training:===')  
            #Ablation experiments Model training
            # brain_machine = importlib.import_module(f"algorithm.RL.brain_TBHL") 
            # address=f"{sys.path[0]}/sequencing_models/TBHL_{rule}_{scenario_id}.pt"         
            # print(f'\n=== scenario={scenario_id},Algorithm TBHL_{rule}model training:===')       
            
            env = simpy.Environment()  
            spf = ShopFloor(env, job_numbers, scenario['parameters'],brain_machine,
                            train=True,address=address)
            spf.simulation() 
                        
        