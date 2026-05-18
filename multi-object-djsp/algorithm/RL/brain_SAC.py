# brain_SAC.py
import random 
import numpy as np 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import agent.sequencing as sequencing
import common.multiobject as mutilobjectivemanager
from common.shared_modules import ContinuousSequencingBrain

class sequencing_brain(ContinuousSequencingBrain):
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initializer
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # ========== SAC-specific parameters ==========
        self.tau = 0.005  # Target network soft update
        self.target_entropy_coeff = 0.98  # Target entropy coefficient
        self.trajectory_buffer_size = 1280  # Replay buffer size
        self.batch_size = 128  # SAC batch size
        
        # ========== Network initialization ==========
        if self.train == True: 
            self.network = SACNetwork(self.input_size)
            self.algorithm_name = "SAC"
            self.training_epochs = 5  # SAC training epochs
            self.env.process(self.training_process_sac())
            self.env.process(self.warm_up_process())
            
        if self.train == False:         
            self.network = SACNetwork(self.input_size)
            self.network.load_state_dict(torch.load(self.address))            
            self.network.eval()  
            self.multi_obj_manager.get_per_baselines(self.address)
            
    # ========== SAC training process ==========
    def training_process_sac(self):  
        """SAC training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting SAC training...")
        train_steps = 0
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodic training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > 0.1 * self.trajectory_buffer_size :
                self.samples -= 0.1 * self.trajectory_buffer_size              
                actor_loss, critic_loss, alpha_loss = self.train_sac()                
                # Track losses
                if actor_loss is not None:
                    self.rule_loss_record.append(actor_loss)  
                    self.value_loss_record.append(critic_loss)  
                    self.loss_record.append(alpha_loss) 
                    
                    # Print training info periodically
                    if train_steps % 10 == 0:
                        current_alpha = self.network.log_alpha.exp().item()
                        print(f'[SAC Training] Steps:{train_steps}, '
                            f'WIP:{self.job_creator.in_system_job_no}, '
                            f'Arrived jobs:{self.job_creator.index_jobs}, '
                            f'Actor loss:{actor_loss:.4f}, '
                            f'Critic loss:{critic_loss:.4f}, '
                            f'Alpha loss:{alpha_loss:.4f} ')
            
            train_steps += 1
            yield self.env.timeout(100) 
         
        # Save model
        address = self.address.format(sys.path[0])
        torch.save(self.network.state_dict(), address)
        pref_address = address.replace('.pt', '_preferences.pkl')
        self.multi_obj_manager.save_training_results(pref_address)
        print(f"SAC model saved: {address}")
   
    def train_sac(self):
        """SAC training function"""
        if len(self.trajectory_buffer) < self.batch_size:
            return None, None, None
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_alpha_loss = 0
        
        # Multi-epoch training
        for epoch in range(self.training_epochs):
            # Sample from replay buffer
            batch = self.sample_sac_batch(self.batch_size)
            
            if batch is None:
                continue
            
            # Convert to tensors
            states = torch.FloatTensor(batch['states']).to(self.network.device)
            actions = torch.FloatTensor(batch['actions']).to(self.network.device)
            rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.network.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.network.device)
            preferences = torch.FloatTensor(batch['preferences']).to(self.network.device)
            
            # SAC update step
            actor_loss, critic_loss, alpha_loss = self.network.update_parameters(
                states, actions, rewards, next_states, preferences
            )
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_alpha_loss += alpha_loss
        
        # Average losses
        avg_actor_loss = total_actor_loss / self.training_epochs
        avg_critic_loss = total_critic_loss / self.training_epochs
        avg_alpha_loss = total_alpha_loss / self.training_epochs
        
        return avg_actor_loss, avg_critic_loss, avg_alpha_loss
    
    def sample_sac_batch(self, batch_size):
        """Sample a batch for SAC training"""
        if len(self.trajectory_buffer) < batch_size:
            return None
        
        batch = random.sample(self.trajectory_buffer, batch_size)
        
        # Initialize lists
        states_list, actions_list, rewards_list = [], [], []
        next_states_list, preferences_list = [], []
        
        for experience in batch:
            # Expected format: (state, action, reward, next_state, preference)
            if len(experience) >= 5:
                state, action, reward, next_state, preference = experience[:5]
                
                # Convert to numpy
                if torch.is_tensor(state):
                    states_list.append(state.detach().cpu().numpy())
                else:
                    states_list.append(state)
                    
                if torch.is_tensor(action):
                    actions_list.append(action.detach().cpu().numpy())
                else:
                    actions_list.append(action)
                    
                rewards_list.append(reward)
                
                if torch.is_tensor(next_state):
                    next_states_list.append(next_state.detach().cpu().numpy())
                else:
                    next_states_list.append(next_state)
                    
                if torch.is_tensor(preference):
                    preferences_list.append(preference.detach().cpu().numpy())
                else:
                    preferences_list.append(preference)
        
        # Convert to numpy arrays
        return {
            'states': np.array(states_list, dtype=np.float32),
            'actions': np.array(actions_list, dtype=np.float32),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'next_states': np.array(next_states_list, dtype=np.float32),
            'preferences': np.array(preferences_list, dtype=np.float32)
        }
    
 
# ========== SAC network architecture ==========
class SACNetwork(nn.Module):
    def __init__(self, input_size, preference_size=3):
        super(SACNetwork, self).__init__()
        
        # ========== Hyperparameters ==========
        self.lr = 0.0003  # SAC often uses a slightly higher LR
        self.input_size = input_size
        self.preference_size = preference_size
        self.hidden_size = 256  # SAC often uses a larger network
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== Feature groups and normalization (reused) ==========
        self.base_info_size = 4
        self.global_info_size = 4
        self.pt_info_size = 3
        self.ttd_slack_info_size = 4
        self.heterogeneity_info_size = 3

        self.norm_base = nn.Sequential(nn.LayerNorm(4), nn.Flatten())
        self.norm_global = nn.Sequential(nn.LayerNorm(4), nn.Flatten())
        self.norm_pt = nn.Sequential(nn.LayerNorm(3), nn.Flatten())
        self.norm_ttd_slack = nn.Sequential(nn.LayerNorm(4), nn.Flatten())
        self.norm_heterogeneity = nn.Sequential(nn.LayerNorm(3), nn.Flatten())
        
        # ========== 1. Shared feature extractor ==========
        self.feature_extractor = nn.Sequential(
            nn.Linear(18, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        
        self.pref_processor = nn.Sequential(
            nn.Linear(self.preference_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # ========== 2. Critic networks (two Q nets) ==========
        # Critic 1
        self.critic1 = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),  # State + preference + action
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Critic 2
        self.critic2 = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # ========== 3. Actor network ==========
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size + 64, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        
        # Output mean and std for Gaussian policy
        self.mean_linear = nn.Linear(self.hidden_size, 3)
        self.log_std_linear = nn.Linear(self.hidden_size, 3)
        
        # ========== 4. Target Critic networks ==========
        self.critic1_target = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.critic2_target = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Copy weights to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # ========== 5. Temperature (auto-tuned) ==========
        self.target_entropy = -torch.prod(torch.Tensor([3]).to(self.device)).item()  # Action dim = 3
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        
        # ========== 6. Optimizers ==========
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr
        )
        
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + 
            list(self.mean_linear.parameters()) + 
            list(self.log_std_linear.parameters()), 
            lr=self.lr
        )
        
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        
        # ========== Initialize weights ==========
        self._init_weights()
        self.to(self.device)
            
    def _init_weights(self):
        """Initialize network weights"""
        for module in [self.feature_extractor, self.pref_processor, 
                      self.critic1, self.critic2, self.actor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0)
        
        # Special init for output layers
        nn.init.orthogonal_(self.mean_linear.weight, gain=0.01)
        nn.init.constant_(self.mean_linear.bias, 0)
        nn.init.orthogonal_(self.log_std_linear.weight, gain=0.01)
        nn.init.constant_(self.log_std_linear.bias, 0)

    def extract_features(self, state_features, preference_vector):
        """Extract shared features"""
        # Feature group normalization
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
        
        # Extract state features
        state_features = self.feature_extractor(normalized_features)
        
        # Process preference features
        pref_features = self.pref_processor(preference_vector)
        
        return state_features, pref_features
    
    def sample_action(self, state_features, preference_vector, deterministic=False):
        """Sample action (training/exploration)"""
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features], dim=-1)
        
        x = self.actor(combined_features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp std
        std = log_std.exp()
        
        if deterministic:
            # Deterministic: use mean
            action = torch.softmax(mean, dim=-1)
        else:
            # Stochastic: sample from Gaussian
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.softmax(x_t, dim=-1)
            
        return action
    
    def evaluate(self, state_features, preference_vector, action):
        """Evaluate action (log_prob and entropy)"""
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features], dim=-1)
        
        x = self.actor(combined_features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Gaussian distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterized sampling
        x_t = normal.rsample()
        action_sample = torch.softmax(x_t, dim=-1)
        
        # log_prob with softmax Jacobian
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        
        # Adjust for softmax transform
        log_prob -= torch.log(1 - action_sample.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # Entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return action_sample, log_prob, entropy
    
    def compute_q_values(self, state_features, preference_vector, action):
        """Compute Q-values"""
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features, action], dim=-1)
        
        q1 = self.critic1(combined_features)
        q2 = self.critic2(combined_features)
        
        return q1, q2
    
    def compute_target_q_values(self, state_features, preference_vector, action):
        """Compute target Q-values"""
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features, action], dim=-1)
        
        q1 = self.critic1_target(combined_features)
        q2 = self.critic2_target(combined_features)
        
        return q1, q2
    
    def update_parameters(self, states, actions, rewards, next_states, preferences):
        """SAC parameter update"""
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        with torch.no_grad():
            # Sample next action from policy
            next_action, next_log_prob, _ = self.evaluate(next_states, preferences, None)
            
            # Compute target Q-values
            target_q1, target_q2 = self.compute_target_q_values(next_states, preferences, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            # TD target
            next_q_value = rewards + 0.99 * target_q
        
        # Update Critic
        current_q1, current_q2 = self.compute_q_values(states, preferences, actions)
        critic1_loss = F.mse_loss(current_q1, next_q_value)
        critic2_loss = F.mse_loss(current_q2, next_q_value)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update Actor
        new_action, log_prob, _ = self.evaluate(states, preferences, None)
        q1_new, q2_new = self.compute_q_values(states, preferences, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.mean_linear.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.log_std_linear.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target, 0.005)
        self.soft_update(self.critic2, self.critic2_target, 0.005)
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update target networks"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def forward(self, state_features, preference_vector):
        """Forward pass (compatibility)"""
        action = self.sample_action(state_features, preference_vector, deterministic=True)
        
        # Value estimate from Critic
        with torch.no_grad():
            q1, q2 = self.compute_q_values(state_features, preference_vector, action)
            value = torch.min(q1, q2)
        
        return action