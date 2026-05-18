# brain_Rainbow_DQN_simplified.py
import random 
import numpy as np 
import sys 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.shared_modules import DiscreteSequencingBrain, DiscreteSchedulingNetwork
from common.cfunctions import state_multi_channel, sequencing_data_generation

# ========== Prioritized Experience Replay Buffer ==========
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        
    def add(self, experience, priority=None):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        if priority is None:
            priority = max_priority
            
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None
            
        probs = self.priorities[:len(self.buffer)]
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha


# ========== Noisy Linear Layer ==========
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init)) if bias else None
        
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)
            
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.epsilon_weight = torch.outer(epsilon_out, epsilon_in)
        if self.bias is not None:
            self.epsilon_bias = epsilon_out
            
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
        
    def forward(self, x):
        if self.training:
            weight = self.weight + self.sigma_weight * self.epsilon_weight
            bias = self.bias
            if bias is not None:
                bias = bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.weight
            bias = self.bias
            
        return F.linear(x, weight, bias)


class sequencing_brain(DiscreteSequencingBrain):
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initializer
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # Rainbow DQN-specific parameters
        self.n_step = 3  # Multi-step learning steps
        self.use_noisy_networks = True  # Use noisy networks
        self.use_dueling_network = True  # Use dueling network
        
       
            
        # Network initialization
        if self.train == True: 
            self.input_size = len(state_multi_channel(self, sequencing_data_generation(self.m_list[0])))  
            self.network = RainbowNetwork(self.input_size)
            self.target_network = RainbowNetwork(self.input_size)
            self.target_network.load_state_dict(self.sequencing_action_NN.state_dict())
            self.target_network.eval()
            self.env.process(self.training_process_parameter_sharing())
            self.env.process(self.update_learning())
            self.env.process(self.warm_up_process())
            
        if self.train == False:         
            self.input_size = len(state_multi_channel(self, sequencing_data_generation(self.m_list[0])))
            self.network = RainbowNetwork(self.input_size)
            self.network.load_state_dict(torch.load(self.address), strict=False)            
            self.network.eval() 
            self.multi_obj_manager.get_per_baselines(self.address) 
            for m in self.m_list:
                if self.ablation == "Ablation1" or self.ablation == "Ablation2":
                    m.job_sequencing = self.action_ablation
                else:
                    m.job_sequencing = self.action_sqc_rule  
  
    def _get_action_probabilities(self, s_t_reshaped, preference, queue_size):
        """Action selection strategy for Rainbow DQN"""
        if self.use_noisy_networks and self.train:
            # Noisy network adds noise automatically during training
            job_probs = self.sequencing_action_NN.forward(s_t_reshaped, preference.reshape(1, 3)).squeeze(0).detach().cpu().numpy()
        elif random.random() < self.epsilon or self.ablation == "Ablation3":
            # Backup exploration strategy
            job_probs = np.random.dirichlet([1] * min(queue_size, self.max_queue_size))
            if len(job_probs) < self.max_queue_size:
                job_probs = np.pad(job_probs, (0, self.max_queue_size - len(job_probs)), 'constant')
        else:         
            job_probs = self.sequencing_action_NN.forward(s_t_reshaped, preference.reshape(1, 3)).squeeze(0).detach().cpu().numpy()
        return job_probs
    
    def training_process_parameter_sharing(self): 
        yield self.env.timeout(self.warm_up + 1)
        # After warm-up, use warm-up experience for supervised learning
        if len(self.trajectory_buffer) > self.minibatch_size:
            for i in range(10):  
                self._train_from_replay()
        
        # Start Rainbow DQN training
        while self.job_creator.in_system_job_no >= 1:
            if self.samples >= self.minibatch_size:
                self._train_from_replay()
                self.samples -= self.minibatch_size
            
            if self.training_step_count % self.target_update_freq == 0:
                self.update_target_network()
            
            # Rainbow DQN: periodically reset noise in noisy networks
            if self.use_noisy_networks and self.training_step_count % 10 == 0:
                self.sequencing_action_NN.reset_noise()
                self.target_network.reset_noise()
                
            yield self.env.timeout(100)
    
    def _train_from_replay(self, num_training_rounds=5):
        """Rainbow DQN training method"""
        
        if len(self.trajectory_buffer) < self.minibatch_size * num_training_rounds:
            num_training_rounds = max(1, len(self.trajectory_buffer) // self.minibatch_size)
            if num_training_rounds == 0:
                return
        
        total_loss = 0.0
        total_samples_used = 0
        
        for round_idx in range(num_training_rounds):
           
            minibatch = random.sample(self.trajectory_buffer, self.minibatch_size)
            indices = None
            weights_tensor = torch.ones(self.minibatch_size, 1, dtype=torch.float32)
            
            total_samples_used += self.minibatch_size
            
            # Extract data
            states = torch.stack([data[0] for data in minibatch], dim=0)
            actions = torch.stack([data[1] for data in minibatch], dim=0)
            rewards = torch.tensor([data[2] for data in minibatch], dtype=torch.float32).unsqueeze(1)
            
            # For next states, if it's a terminal state (None), we can use a zero vector or the current state as a placeholder
            next_states_list = []
            for data in minibatch:
                if data[3] is not None:
                    next_states_list.append(data[3])
                else:
                    next_states_list.append(data[0])
            next_states = torch.stack(next_states_list, dim=0)
            
            preferences = torch.stack([data[4] for data in minibatch], dim=0)
            job_weights = [data[5] for data in minibatch]
            
            # Rainbow DQN Training Logic
            current_q_values = self.sequencing_action_NN.forward(states, preferences)
            
            # Calculate current state-action values for the actions taken in the batch
            current_values_tensor = torch.zeros(self.minibatch_size, 1, dtype=torch.float32)
            for i in range(self.minibatch_size):
                weights_i = job_weights[i]
                _, value = self.selection_job_reward(weights_i, current_q_values[i])
                current_values_tensor[i, 0] = value
            
            # Action selection by current network, value evaluation by target network
            with torch.no_grad():
                # The best action for the network to take in the next state
                next_q_values_current = self.sequencing_action_NN.forward(next_states, preferences)
                next_best_actions = torch.argmax(next_q_values_current, dim=1)
                
                # The value of the best action in the next state according to the target network
                next_q_values_target = self.target_network.forward(next_states, preferences)
                next_q_values_selected = next_q_values_target.gather(1, next_best_actions.unsqueeze(1))
            
            # Calculate target values using multi-step returns
            target_values = rewards + (self.discount_factor ** self.n_step) * next_q_values_selected
            
            # Calculate TD errors and loss
            td_errors = target_values - current_values_tensor
            loss = (weights_tensor * F.smooth_l1_loss(current_values_tensor, target_values, reduction='none')).mean()
            
            # Backpropagation
            self.sequencing_action_NN.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sequencing_action_NN.parameters(), max_norm=5.0)
            self.sequencing_action_NN.optimizer.step()    
            
            total_loss += loss.item()
            
        self.training_step_count += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_training_rounds if num_training_rounds > 0 else 0
        
        # Record training information
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(avg_loss)
        
       
        avg_reward = rewards.mean().item() 
        avg_td_error = td_errors.abs().mean().item() 
        
        print(f'[Rainbow DQN training] '
                f'Steps:{self.training_step_count:04d}, '
                f'In-Process Jobs:{self.job_creator.in_system_job_no}, '
                f'Arrived Jobs:{self.job_creator.index_jobs}, '
                f'Average Loss:{avg_loss:.6f}, '
                f'TD Error:{avg_td_error:.4f}, '
                f'Average Reward:{avg_reward:.4f}')
        
        return avg_loss


class RainbowNetwork(DiscreteSchedulingNetwork):
    """Rainbow DQN network architecture"""
    
    def __init__(self, input_size, max_queue_size=5, preference_size=3, use_noisy=True, use_dueling=True):
        self.use_noisy = use_noisy
        self.use_dueling = use_dueling
        super().__init__(input_size, max_queue_size, preference_size)
    
    def _init_network_layers(self):
        """Initialize Rainbow network layers"""
        LinearLayer = NoisyLinear if self.use_noisy else nn.Linear
        
        # 1. State feature extraction
        self.state_extractor = nn.Sequential(
            LinearLayer(18, 20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(20),
            nn.Dropout(0.1),
            LinearLayer(20, 16),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            LinearLayer(16, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1)
        )
        
        # 2. Preference feature enhancement
        self.pref_enhancer = nn.Sequential(
            LinearLayer(self.preference_size, 6),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(6),
            nn.Dropout(0.1),
            LinearLayer(6, self.preference_size),
            nn.Softmax(dim=-1)
        )
        
        # 3. Feature fusion
        self.feature_fusion = nn.Sequential(
            LinearLayer(13, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(10),
            nn.Dropout(0.1),
            LinearLayer(10, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            LinearLayer(8, 6),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.residual_transform = LinearLayer(13, 6)
        
        # 4. Dueling network architecture
        if self.use_dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                LinearLayer(6 + self.preference_size, 16),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(16),
                nn.Dropout(0.1),
                LinearLayer(16, 8),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(8),
                LinearLayer(8, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                LinearLayer(6 + self.preference_size, 16),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(16),
                nn.Dropout(0.1),
                LinearLayer(16, 12),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(12),
                nn.Dropout(0.1),
                LinearLayer(12, self.max_queue_size)
            )
        else:
            # Standard Q-network
            self.q_value_head = nn.Sequential(
                LinearLayer(6 + self.preference_size, 16),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(16),
                nn.Dropout(0.1),
                LinearLayer(16, 12),
                nn.LeakyReLU(negative_slope=0.1),
                nn.LayerNorm(12),
                nn.Dropout(0.1),
                LinearLayer(12, self.max_queue_size)
            )
        
        # 5. Probability output head
        self.probability_head = nn.Sequential(
            nn.Softmax(dim=-1)
        )
        
        # 6. Direct preference path
        self.pref_direct_q = nn.Sequential(
            LinearLayer(self.preference_size, 8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            LinearLayer(8, self.max_queue_size)
        )
        
        # 7. Fusion gate parameters
        self.fusion_gate = nn.Sequential(
            LinearLayer(6, 3),
            nn.LeakyReLU(negative_slope=0.1),
            LinearLayer(3, 1),
            nn.Sigmoid()
        )
    
    def reset_noise(self):
        """Reset noise in noisy networks (only if using noisy networks)"""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
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
        
        # 5. Dueling or standard Q-network
        if self.use_dueling:
            # Dueling network: value stream + advantage stream
            value = self.value_stream(torch.cat([fused_features, enhanced_pref], dim=-1))
            advantage = self.advantage_stream(torch.cat([fused_features, enhanced_pref], dim=-1))
            
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard Q-network
            q_values_from_fusion = self.q_value_head(
                torch.cat([fused_features, enhanced_pref], dim=-1)
            )
            
            # Direct preference path
            q_values_from_pref = self.pref_direct_q(enhanced_pref)
            
            # Adaptive fusion
            gate = self.fusion_gate(fused_features)
            q_values = gate * q_values_from_fusion + (1 - gate) * q_values_from_pref
        
        # 6. Convert Q-values to selection probabilities
        job_probabilities = self.probability_head(q_values)
        
        return job_probabilities