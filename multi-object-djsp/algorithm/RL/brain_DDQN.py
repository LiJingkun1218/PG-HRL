# brain_DDQN_simplified.py
import random 
import numpy as np 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.shared_modules import DiscreteSequencingBrain, DiscreteSchedulingNetwork
from common.cfunctions import state_multi_channel, sequencing_data_generation

class sequencing_brain(DiscreteSequencingBrain):
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initializer
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        # DDQN-specific parameters
        self.use_prioritized_replay = False
        self.use_noisy_networks = False
        self.use_dueling_network = False
        # Network initialization
        if self.train == True:
            self.input_size = len(state_multi_channel(self, sequencing_data_generation(self.m_list[0])))
            self.network = SchedulingNetwork(self.input_size)
            self.target_network = SchedulingNetwork(self.input_size)
            self.target_network.load_state_dict(self.sequencing_action_NN.state_dict())
            self.target_network.eval()
            self.env.process(self.training_process_parameter_sharing())
            self.env.process(self.update_learning())
            self.env.process(self.warm_up_process())
        if self.train == False:
            self.input_size = len(state_multi_channel(self, sequencing_data_generation(self.m_list[0])))
            self.network = SchedulingNetwork(self.input_size, max_queue_size=self.max_queue_size)
            self.network.load_state_dict(torch.load(self.address))
            self.network.eval()
            self.multi_obj_manager.get_per_baselines(self.address)
            for m in self.m_list:
                if self.ablation == "Ablation1" or self.ablation == "Ablation2":
                    m.job_sequencing = self.action_ablation
                else:
                    m.job_sequencing = self.action_sqc_rule
    
    def _get_action_probabilities(self, s_t_reshaped, preference, queue_size):
        """Action selection strategy for DDQN"""
        if random.random() < self.epsilon or self.ablation == "Ablation3":
            job_probs = np.random.dirichlet([1] * min(queue_size, self.max_queue_size))
            if len(job_probs) < self.max_queue_size:
                job_probs = np.pad(job_probs, (0, self.max_queue_size - len(job_probs)), 'constant')
        else:         
            job_probs = self.sequencing_action_NN.forward(s_t_reshaped, preference.reshape(1, 3)).squeeze(0).detach().cpu().numpy()
        return job_probs
    
    def _train_from_replay(self, num_training_rounds=5):
        """DDQN training method"""
        if len(self.trajectory_buffer) < self.minibatch_size * num_training_rounds:
            num_training_rounds = max(1, len(self.trajectory_buffer) // self.minibatch_size)
            if num_training_rounds == 0:
                return
        
        total_loss = 0.0
        total_samples_used = 0
        
        for round_idx in range(num_training_rounds):
            if len(self.trajectory_buffer) < self.minibatch_size:
                break
                
            minibatch = random.sample(self.trajectory_buffer, self.minibatch_size)
            total_samples_used += self.minibatch_size
            
            # Prepare training data
            states = torch.stack([data[0] for data in minibatch], dim=0)
            rewards = torch.tensor([data[2] for data in minibatch], dtype=torch.float32)
            preferences = torch.stack([data[4] for data in minibatch], dim=0)        
            job_weights = [data[5] for data in minibatch]
            
            # Handle the next states for DDQN
            next_states_list = []
            for data in minibatch:
                if data[3] is not None:
                    next_states_list.append(data[3])
                else:
                    next_states_list.append(data[0])
            next_states = torch.stack(next_states_list, dim=0)
            
            # DDQN training logic
            current_probs = self.sequencing_action_NN.forward(states, preferences)
        
            current_values_tensor = torch.zeros(self.minibatch_size, dtype=torch.float32)
            for i in range(self.minibatch_size):
                weights_i = job_weights[i]
                _, value = self.selection_job_reward(weights_i, current_probs[i])
                current_values_tensor[i] = value
            
            with torch.no_grad():
                target_next_probs = self.target_network.forward(next_states, preferences)
                target_next_values = torch.zeros(self.minibatch_size, dtype=torch.float32)
                for i in range(self.minibatch_size):
                    next_weights_i = job_weights[i]  # Note: The weight of the next state should be used here
                    _, target_value = self.selection_job_reward(next_weights_i, target_next_probs[i])
                    target_next_values[i] = target_value
            
            target_next_values_tensor = torch.tensor(target_next_values, dtype=torch.float32)
            target_values = rewards + self.discount_factor * target_next_values_tensor
            
            loss = F.smooth_l1_loss(current_values_tensor, target_values)
            
            self.sequencing_action_NN.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sequencing_action_NN.parameters(), max_norm=5.0)
            self.sequencing_action_NN.optimizer.step()
            
            total_loss += loss.item()
            
            if self.training_step_count % 100 == 0:
                self.update_target_network()
            
            self.training_step_count += 1
        
        avg_loss = total_loss / num_training_rounds if num_training_rounds > 0 else 0
        
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(avg_loss)
        
        avg_reward = rewards.mean().item() if len(rewards) > 0 else 0
        
       
        print(f'[DDQN Training] '
                f'Steps:{self.training_step_count:04d}, '
                f'In-Process Jobs:{self.job_creator.in_system_job_no}, '
                f'Arrived Jobs:{self.job_creator.index_jobs}, '
                f'Average Loss:{avg_loss:.6f}, '
                f'Average Reward:{avg_reward:.4f}')
        
        return avg_loss


class SchedulingNetwork(DiscreteSchedulingNetwork):
    """DDQN network architecture"""
    
    def __init__(self, input_size, max_queue_size=5, preference_size=3):
        super().__init__(input_size, max_queue_size, preference_size)
    
    def _init_network_layers(self):
        """Initialize DDQN network layers"""
        # 1. State feature extraction
        self.state_extractor = nn.Sequential(
            nn.Linear(18, 20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(20),
            nn.Dropout(0.1),
            nn.Linear(20, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1)
        )
        
        # 2. Preference feature enhancement
        self.pref_enhancer = nn.Sequential(
            nn.Linear(self.preference_size, 6),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(6),
            nn.Dropout(0.1),
            nn.Linear(6, self.preference_size),
            nn.Softmax(dim=-1)
        )
        
        # 3. Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(13, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1),
            nn.Linear(10, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, 6),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_transform = nn.Linear(13, 6)
        
        # 4. Q-value output head
        self.q_value_head = nn.Sequential(
            nn.Linear(6 + self.preference_size, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(12),
            nn.Dropout(0.1),
            nn.Linear(12, self.max_queue_size)
        )
        
        # 5. Probability output head
        self.probability_head = nn.Sequential(
            nn.Softmax(dim=-1)
        )
        
        # 6. Direct preference path
        self.pref_direct_q = nn.Sequential(
            nn.Linear(self.preference_size, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Linear(8, self.max_queue_size)
        )
        
        # 7. Fusion gate parameters
        self.fusion_gate = nn.Sequential(
            nn.Linear(6, 3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state_features, preference_vector):
        # 1. Feature group normalization
        base_info = state_features[:, :4]
        global_info = state_features[:, 4:8]
        pt_info = state_features[:, 8:11]
        ttd_slack_info = state_features[:, 11:15]
        heterogeneity_info = state_features[:, 15:18]
        
        # Apply normalization
        base_norm = self.norm_base(base_info)
        global_norm = self.norm_global(global_info)
        pt_norm = self.norm_pt(pt_info)
        ttd_slack_norm = self.norm_ttd_slack(ttd_slack_info)
        heterogeneity_norm = self.norm_heterogeneity(heterogeneity_info)
        
        # Concatenate normalized features
        normalized_features = torch.cat([
            base_norm, global_norm, pt_norm, 
            ttd_slack_norm, heterogeneity_norm
        ], dim=-1)
        
        # 2. State feature extraction
        state_features = self.state_extractor(normalized_features)
        
        # 3. Preference feature enhancement
        enhanced_pref = self.pref_enhancer(preference_vector)
        
        # 4. Feature fusion (with residual connection)
        combined_features = torch.cat([state_features, enhanced_pref], dim=-1)
        fusion_out = self.feature_fusion(combined_features)
        residual = self.residual_transform(combined_features)
        
        # True residual connection
        fused_features = fusion_out + residual
        
        # 5. Dual-path Q-value calculation
        q_values_from_fusion = self.q_value_head(
            torch.cat([fused_features, enhanced_pref], dim=-1)
        )
        
        q_values_from_pref = self.pref_direct_q(enhanced_pref)
        
        # 6. Adaptive fusion
        gate = self.fusion_gate(fused_features)
        q_values = gate * q_values_from_fusion + (1 - gate) * q_values_from_pref
        
        # 7. Convert Q-values to selection probabilities
        job_probabilities = self.probability_head(q_values)
        
        return job_probabilities