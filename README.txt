Code User Manual (Full Version)


**Applicable to the paper**：\[Adaptive Preference-Guided Hybrid Deep Reinforcement Learning with Mathematical Programming for Multi-Objective Just-in-Time Scheduling]



1. Environment Dependencies
	All required third-party libraries are listed in `requirements.txt`.
		*pip install -r requirements.txt*

2. Project Structure Overview
project\_root/
│
├── agent/                              # Simulation agents
│   ├── agent\_machine.py                # Machine model (queue, processing, delay)
│   ├── job\_creation.py                 # Job generator (dynamic arrival, processing time, due date)
│   └── sequencing.py                   # Dispatching rule library (CR, MDD, ATC, SPT, FIFO, 20+ rules)
│
├── algorithm/                          # Algorithm implementations
│   ├── RL/                             # DRL algorithms
│   │   ├── brain\_PG_HRL.py               # Proposed PG_HRL algorithm
│   │   ├── brain\_SAC.py                 
│   │   ├── brain\_TD3.py                
│   │   ├── brain\_DDQN.py                
│   │   └── brain\_Rainbow\_DQN.py        
│   │
│   └── SI/                             # Multi-objective evolutionary comparison algorithms
│       ├── DMOA\_MHKT.py       
│       ├── FCP.py                      
│       ├── KT\_DMOEA.py                 
│       └── Tr\_DMOEA.py                 
│
├── common/                             # Shared modules
│   ├── cfunctions.py                   # Core scheduling functions (state update, decoding, objective calculation)
│   ├── experiment\_scene.py             # Orthogonal experiment scenario definitions (9 L9 scenarios)
│   ├── experimental\_analysis.py        # Experiment analysis system (statistical tests, visualization, reporting)
│   ├── multiobject.py                  # Multi-objective manager (preference learning, Pareto archive)
│   └── shared\_modules.py               # RL base classes, simulation environment, PD-MORL components
│
├── test\_result/                        # Experiment result output (auto-created)
│
├── sequencing\_models/                       # .pt model files and preference baselines (.pkl)│ 
│
├── train\_rl\_model.py                  # DRL model training entry
├── compare\_dispatching\_rules.py                   # Dispatching rule comparison experiment
├── compare\_multi\_objective\_algorithms.py                    # Multi-objective evolutionary algorithm comparison
├── generalization\_experiment.py                     # Generalization capability test (cross-scenario)
├── compare\_rl\_algorithms.py                  # DRL algorithm horizontal comparison
└── ablation\_experiment.py                      # Ablation experiment

3. Quick Start: Training the PG_HRL Model
3.1 Single-Scenario Training
	Default configuration: The PG_HRL algorithm is trained on all 9 orthogonal scenarios. To modify this, edit the **benchmark** and **job_numbers** variables in **train_rl_model.py**.
	Custom training commands** (directly modify the script or pass parameters; argparse is not implemented in the current script, so modifying variables directly is recommended):
	(1)Modify **benchmark = ['PG-HRL']** to select the algorithm.
	(2)Modify **job_numbers = 10000** to control simulation length.
	(3)Trained models are saved at **sequencing_models/PG_HRL\_{scenario\_id}.pt**.
3.2 Training Other RL Algorithms (SAC, TD3, DDQN, Rainbow\_DQN)
	Modify the following in **train_rl_model.py**:
		benchmark = \['SAC']  # or 'TD3', 'DDQN', 'Rainbow\_DQN'
	Then run the script. Note that different algorithms require longer training times (especially continuous action algorithms such as SAC/TD3).

4. Running Comparison Experiments
	All comparison experiment scripts automatically output results to corresponding subdirectories under **test_result/**, and generate HV and CR metrics, which are saved in **\_metrics.xlsx** files.
4.1 Dispatching Rule Comparison (6 Rules)
	Run the dispatching rule comparison:
		compare_dispatching_rules.py
	(1)Run parameters: **run\num = 25** (number of repetitions), **job_num = 50** (number of jobs)
	(2)Compared rules: **['CR', 'MDD', 'MOD', 'MS', 'ATC', 'EDD']** (can be extended in the script)
	(3)Output directory: **test_result/Rule/**
	(4)Each run generates **{scenario_id}_metrics.xlsx**, which contains two sheets for HV and CR.
 4.2 Multi-Objective Evolutionary Algorithm Comparison (DMOA-MHKT, FCP, KT-DMOEA, Tr-DMOEA)
	Run the dispatching rule comparison:
		compare_multi_objective_algorithms.py
	(1)Compared algorithms: **benchmark = ['KT_DMOEA', 'DMOA_MHKT', 'FCP', 'Tr_DMOEA']**
	(2)Each algorithm runs independently and generates the Pareto front
	(3)Output directory: **test_result/DMO/**
	(4)Note: This script first uses the pre-trained PG_HRL model (must be trained in advance) to generate the DRL baseline, then runs the evolutionary algorithms. Please ensure that **sequencing_models/PG_HRL\_{scenario\_id}.pt** exists, otherwise the DRL part will fail.
4.3 Reinforcement Learning Algorithm Comparison (PG-HRL vs. SAC, TD3, DDQN, Rainbow_DQN)
	Run the dispatching rule comparison:
		compare_rl_algorithms.py*
	(1)Compared algorithms: benchmark = **['SAC', 'TD3', 'PG-HRL', 'DDQN', 'Rainbow\_DQN']**
	(2) Automatically loads pre-trained models for each algorithm (if not available, they must be trained in advance using **train_rl_model.py**)
	(3)Output directory: **test_result/Ref_Learning/**
 4.4 Generalization Experiment (Cross-Scenario Testing of PG_HRL's Generalization Ability)
	Run the dispatching rule comparison:
		generalization_experiment.py
	(1)Train models using three representative training scenarios (L9-6, L9-8, L9-4), then test on all 9 scenarios
	(2)Output directory: \*\*test\_result/Robust/\*\*
	(3)Generated outputs:
		Generalization bar charts, heatmaps, and radar charts (**.png** and **.pdf**)
		Detailed analysis Excel (model comparison per scenario, inter-group comparison, overall model performance)
		Raw test data (**.pkl**)
 4.5 Ablation Study (Verifying the Effectiveness of PG-HRL Modules)
	Run the dispatching rule comparison:
		ablation_experiment.py
	(1)Ablation settings:
		Ablation1: Remove the first operation delay calculation module
		Ablation2: Remove the weight optimization module
		Ablation3: Replace preference guidance with random weights
		Ablation4: Replace the proposed preference generation method with that of PD-MORL
	(2)Output directory: **test_result/Ablation/**

5. Experimental Result Analysis and Visualization
	After all experiments are completed, the analysis pipeline in **common/experimental_analysis.py** automatically generates statistical reports and charts.
5.1 Automatic Analysis (Included at the End of Each Comparison Script)
	Each comparison script ends with the following code:
		if out\_analysis:
			data\_analysis\_report(sub\_path='Rule')  # or 'DMO', 'Ref\_Learning', 'Ablation', 'Robust'*
	This function will:
		(1) Load all **_metrics.xlsx** files under **test_result/{sub_path}/**
		(2)Compute the mean and standard deviation of HV and CR, and perform Friedman test + Nemenyi post-hoc test
		(3)Generate detailed scenario indicator tables (with significance markers "**","*", "†"), win rate statistics, and stability-weighted rankings
		(4)Output charts: box plots, radar charts, heatmaps, performance distribution diagrams, etc.
		(5)Save the results to **test_result/experimental_analysis/{sub_path}/**
5.2 Manual Analysis (Optional)
	If you need to re-analyze a specific experiment separately:
		from common.experimental\_analysis import data\_analysis\_report
		data\_analysis\_report(excel\_path='test\_result', sub\_path='Ref\_Learning')

6. Frequently Asked Questions (FAQ) and Solutions
(1)**ModuleNotFoundError: No module named 'agent'**: The project root directory is not added to the Python path------Add **sys.path.append(os.getcwd())** at the beginning of the script, or run **python -m script_name** from the project root directory.
(2)Loss becomes **NaN** during training: Gradient explosion or data anomaly------Reduce the learning rate in **shared_modules.py** (e.g., **lr=0.0001**), or increase the gradient clipping threshold.
(3)Simulation hangs and does not terminate: The machine queue is empty and no new jobs arrive------Check if **initial_job_assignment** in **job_creation.py** runs correctly; ensure the **sufficient_stock** event is triggered.
(4)HV calculation fails due to missing **pymoo**: Optional dependency------Run **pip install pymoo**; if not installed, the program will use a simplified HV approximation (which does not affect comparison trends).|
(5)Multi-objective evolutionary algorithms run slowly: Large population size and number of generations------Reduce **pop_size** and **max_gen** in **compare_multi_objective_algorithms.py** (default values are 100/150).
(6)Generalization experiment cannot find the PG_HRL model: Model not pre-trained------First run **train_rl_model.py** to train models for the three scenarios: L9-6, L9-8, and L9-4.

7. Customizing Experimental Configurations
7.1 Modifying Scenario Parameters
	Edit the **orthogonal_scenarios** list in **common/experiment_scene.py**, or directly pass a parameter dictionary.
7.2 Adjusting DRL Training Hyperparameters
	Modify the following in **ContinuousSequencingBrain** or **DiscreteSequencingBrain** within **common/shared_modules.py**:
	(1)**discount_factor**: Discount factor (default: 0.99)
	(2)**epsilon**: Exploration rate (default: 0.1)
	(3)**trajectory_buffer_size**: Experience replay buffer size
	(4)**minibatch_size**: Mini-batch size
7.3 Changing the Set of Dispatching Rules
	Edit the **benchmark** list in **compare\_dispatching\_rules.py**. You may select any rule function name defined in **agent/sequencing.py**.

8. Output File Description
(1)**sequencing_models/*.pt**: Trained DRL model weights
(2)**sequencing_models/*_preferences.pkl**: Preference baselines generated during training (used for inference)
(3)**test_result/{Rule,DMO,Ref\_Learning,Ablation,Robust}/*_metrics.xlsx**: Raw HV and CR for each experiment
(4)**test_result/experimental_analysis/{sub\_path}/**: Statistical analysis reports (Excel tables, rankings, significance results)
(5)**test_result/Robust/*.png/pdf**: Visualization charts for the generalization experiment
(6)**experiment_analysis.log**: Detailed run logs





