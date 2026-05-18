# brain_TD3.py
import random 
import numpy as np 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.shared_modules import ContinuousSequencingBrain

class sequencing_brain(ContinuousSequencingBrain):
    def __init__(self, env, job_creator, all_machines, job_numbers, *args, **kwargs):
        # Call parent class initializer
        super().__init__(env, job_creator, all_machines, job_numbers, *args, **kwargs)
        
        # TD3-specific parameters
        self.tau = 0.005  # Target network soft update
        self.policy_noise = 0.2  # Target policy noise
        self.noise_clip = 0.5  # Noise clip range
        self.policy_delay = 2  # Policy delay
        self.trajectory_buffer_size = 256  # Replay buffer size
        self.batch_size = 128  # TD3 batch size
        self.total_train_steps = 0  # Total training steps
        
        # Initialize TD3 network
        if self.train == True: 
            self.network = TD3Network(self.input_size)
            self.env.process(self.training_process_td3())  # TD3 training process
        else:
            self.network = TD3Network(self.input_size)
            if self.address:
                self.network.load_state_dict(torch.load(self.address))
            self.network.eval()
    
    def training_process_td3(self):  
        """TD3 training process"""
        yield self.env.timeout(self.warm_up + 1)
        
        print("Starting TD3 training...")
        train_steps = 0
        
        while self.job_creator.in_system_job_no >= 1:
            # Periodic training
            if len(self.trajectory_buffer) >= self.trajectory_buffer_size and self.samples > 0.1 * self.trajectory_buffer_size :
                self.samples -= 0.1 * self.trajectory_buffer_size              
                actor_loss, critic_loss = self.train_td3()
                if actor_loss is not None: 
                    if self.total_train_steps % 10 == 0:
                        print(f'TD3 Training Steps:{self.total_train_steps}, '
                            f'WIP:{self.job_creator.in_system_job_no}, '
                            f'Arrived jobs:{self.job_creator.index_jobs}, '
                            f'Actor loss:{actor_loss:.3f}, '
                            f'Critic loss:{critic_loss:.3f},')
                    
            self.total_train_steps += 1
            yield self.env.timeout(100)  # Check every 100 time units
         
        # Save model
        if self.address:
            address = self.address.format(sys.path[0])
            torch.save(self.network.state_dict(), address)
            pref_address = address.replace('.pt', '_preferences.pkl')
            self.multi_obj_manager.save_training_results(pref_address)
            print(f"TD3 model and preference vectors saved: {address}, {pref_address}")
   
    def train_td3(self):
        """TD3 training function"""
        if len(self.trajectory_buffer) < self.batch_size:
            return None, None
        
        # Sample from replay buffer
        batch = self.sample(self.batch_size)
        if batch is None:
            return None, None
        
        # Convert to tensors
        states = torch.FloatTensor(batch[0]).to(self.network.device)
        actions = torch.FloatTensor(batch[1]).to(self.network.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.network.device)
        next_states = torch.FloatTensor(batch[3]).to(self.network.device)
        preferences = torch.FloatTensor(batch[4]).to(self.network.device)
        
        # TD3 update step
        actor_loss, critic_loss = self.network.update_parameters(
            states, actions, rewards, next_states, preferences,
            self.total_train_steps
        )
        
        return actor_loss, critic_loss
    
    def sample(self, batch_size):
        """Sample a batch, return a list instead of a dict."""
        if len(self.trajectory_buffer) < batch_size:
            return None
        
        batch = random.sample(self.trajectory_buffer, batch_size)
        
        # Initialize lists
        states_list, actions_list, rewards_list = [], [], []
        next_states_list, preferences_list = [], []
        
        for state, action, reward, next_state, preference, w1, w2 in batch:
            states_list.append(state.detach().cpu().numpy())
            actions_list.append(action.detach().cpu().numpy())
            rewards_list.append(reward)
            next_states_list.append(next_state.detach().cpu().numpy())
            preferences_list.append(preference.detach().cpu().numpy())
        
        # Return list
        return [
            np.array(states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.float32),
            np.array(rewards_list, dtype=np.float32).reshape(-1, 1),
            np.array(next_states_list, dtype=np.float32),
            np.array(preferences_list, dtype=np.float32)
        ]


class TD3Network(nn.Module):
    """TD3 network with TD3-specific components."""
    def __init__(self, input_size, preference_size=3):
        super(TD3Network, self).__init__()
        
        # ========== TD3 hyperparameters ==========
        self.lr = 0.0001  # TD3 learning rate
        self.input_size = input_size
        self.preference_size = preference_size
        self.hidden_size = 256
        
        # ========== Device setup ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== Shared feature extraction ==========
        # Feature group normalization
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
        
        # ========== 2. Actor network (deterministic) ==========
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size + 64, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3),  # Output 3D action (preference vector)
        )
        
        # ========== 3. Critic networks (twin Q) ==========
        # Critic 1
        self.critic1 = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
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
        
        # ========== 4. Target networks ==========
        # Target Actor
        self.actor_target = nn.Sequential(
            nn.Linear(self.hidden_size + 64, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3),
        )
        
        # Target Critic 1
        self.critic1_target = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Target Critic 2
        self.critic2_target = nn.Sequential(
            nn.Linear(self.hidden_size + 64 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Copy weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # ========== 5. Optimizers ==========
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=self.lr
        )
        
        # ========== TD3 hyperparameters ==========
        self.tau = 0.005  # Target network soft update
        self.policy_noise = 0.2  # Target policy noise
        self.noise_clip = 0.5  # Noise clip range
        self.policy_delay = 2  # Policy delay
        
        # ========== Weight corruption flag ==========
        self.weight_corrupted = False
        
        # ========== Initialize weights ==========
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.feature_extractor, self.pref_processor, 
                      self.critic1, self.critic2, self.actor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0)
        
        # Copy to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def extract_features(self, state_features, preference_vector):
        """Extract shared features."""
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
    
    def _safe_softmax(self, x, dim=-1, temperature=1.0):
        """Numerically stable softmax."""
        # Check for NaNs
        if torch.isnan(x).any():
            print("⚠️  Softmax input has NaN, returning uniform distribution")
            return torch.ones_like(x) / x.shape[dim]
        
        # Temperature scaling for stability
        x = x / temperature
        
        # Subtract max for stability (key step)
        max_vals = torch.max(x, dim=dim, keepdim=True)[0]
        x_stable = x - max_vals
        
        # Exponentiate
        exp_x = torch.exp(x_stable)
        
        # Normalize (avoid divide-by-zero)
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True) + 1e-8
        output = exp_x / sum_exp
        
        # Check output again
        if torch.isnan(output).any():
            print("⚠️  Softmax output has NaN, returning uniform distribution")
            return torch.ones_like(x) / x.shape[dim]
            
        return output
    
    def forward(self, state_features, preference_vector, mode="policy"):        
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features], dim=-1)
    
        raw_action = self.actor(combined_features)
      
        action = self._safe_softmax(raw_action, dim=-1, temperature=2.0)
        
        if mode == "policy":
            return action
        
        # Compute Q-values
        q1, q2 = self.compute_q_values(state_features, preference_vector, action)
        value = torch.min(q1, q2)
        
        if mode == "value":
            return value
        elif mode == "both":
            return action, value
    
    def compute_q_values(self, state_features, preference_vector, action):
        """Compute Q-values."""
        state_features, pref_features = self.extract_features(state_features, preference_vector)
        combined_features = torch.cat([state_features, pref_features, action], dim=-1)
        
        q1 = self.critic1(combined_features)
        q2 = self.critic2(combined_features)
        
        return q1, q2
    
    def compute_target_q_values(self, next_states, preferences, next_actions):
        """Compute target Q-values (TD3 noisy target policy)."""
        state_features, pref_features = self.extract_features(next_states, preferences)
        combined_features = torch.cat([state_features, pref_features, next_actions], dim=-1)
        
        q1 = self.critic1_target(combined_features)
        q2 = self.critic2_target(combined_features)
        
        return q1, q2
    
    def _check_and_reset_network(self):
        """Check and reset network weights."""
        # Check Actor weights
        for name, param in self.actor.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"🚨 Actor param {name} corrupted! Resetting network...")
                self._reset_network_weights()
                self.weight_corrupted = True
                return True
        
        # Check Critic weights
        for name, param in self.critic1.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"🚨 Critic1 param {name} corrupted! Resetting network...")
                self._reset_network_weights()
                self.weight_corrupted = True
                return True
                
        for name, param in self.critic2.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"🚨 Critic2 param {name} corrupted! Resetting network...")
                self._reset_network_weights()
                self.weight_corrupted = True
                return True
                
        return False
    
    def _reset_network_weights(self):
        """Reset network weights."""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                # Use smaller initialization
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)
        
        # Reset all networks
        self.feature_extractor.apply(init_layer)
        self.pref_processor.apply(init_layer)
        self.actor.apply(init_layer)
        self.critic1.apply(init_layer)
        self.critic2.apply(init_layer)
        
        # Reset target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        print("✅ Network weights reset")
    
    def update_parameters(self, states, actions, rewards, next_states, preferences, total_steps):
        """TD3 parameter update with gradient clipping and NaN checks."""
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        preferences = preferences.to(self.device)
        
        # ========== 1. Input checks ==========
        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"⚠️  {name} has NaN/Inf, skipping batch")
                return False
            return True
        
        # Check all inputs
        if not (check_tensor(states, "states") and 
                check_tensor(actions, "actions") and 
                check_tensor(rewards, "rewards") and 
                check_tensor(next_states, "next_states") and 
                check_tensor(preferences, "preferences")):
            return 0.0, 0.0
        
        # ========== 2. Periodic weight check ==========
        if total_steps % 50 == 0:
            self._check_and_reset_network()
        
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # TD3: target policy noise
            state_features, pref_features = self.extract_features(next_states, preferences)
            combined_features = torch.cat([state_features, pref_features], dim=-1)
            
            # Target policy action (stable softmax)
            raw_next_action = self.actor_target(combined_features)
            next_action = self._safe_softmax(raw_next_action, dim=-1, temperature=2.0)
            
            # Add policy noise (TD3 key improvement)
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = next_action + noise
            
            # Clamp and renormalize action
            next_action = torch.clamp(next_action, 0, 1)
            next_action = next_action / (next_action.sum(dim=1, keepdim=True) + 1e-8)
            
            # Target Q as min of two Critics
            target_q1, target_q2 = self.compute_target_q_values(next_states, preferences, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # TD target
            next_q_value = rewards + 0.99 * target_q
        
        # ========== 3. Update Critic ==========
        current_q1, current_q2 = self.compute_q_values(states, preferences, actions)
        
        # Validate Q-values
        if torch.isnan(current_q1).any() or torch.isnan(current_q2).any() or \
           torch.isnan(next_q_value).any():
            print("⚠️  Q-values have NaN, skipping Critic update")
            return 0.0, 0.0
        
        critic1_loss = F.mse_loss(current_q1, next_q_value)
        critic2_loss = F.mse_loss(current_q2, next_q_value)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        
        self.critic_optimizer.step()
        
        # ========== 4. Delayed Actor update ==========
        actor_loss = None
        if total_steps % self.policy_delay == 0:
            # Actor loss
            state_features, pref_features = self.extract_features(states, preferences)
            combined_features = torch.cat([state_features, pref_features], dim=-1)
            
            # New action (stable softmax)
            raw_new_actions = self.actor(combined_features)
            new_actions = self._safe_softmax(raw_new_actions, dim=-1, temperature=2.0)
            
            # Validate action
            if torch.isnan(new_actions).any():
                print("⚠️  New action has NaN, skipping Actor update")
                return 0.0, critic_loss.item()
            
            q1_new, _ = self.compute_q_values(states, preferences, new_actions)
            
            # Validate Q-values
            if torch.isnan(q1_new).any():
                print("⚠️  New Q-value has NaN, skipping Actor update")
                return 0.0, critic_loss.item()
            
            # Actor maximizes Q
            actor_loss = -q1_new.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            # Actor gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            
            self.actor_optimizer.step()
            
            # ========== 5. Soft update targets ==========
            self.soft_update(self.critic1, self.critic1_target, self.tau)
            self.soft_update(self.critic2, self.critic2_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
        
        return actor_loss.item() if actor_loss is not None else 0.0, critic_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update target networks."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_action(self, state_features, preference_vector, exploration=True):
        """Get action (inference/exploration)."""
        with torch.no_grad():
            action = self.forward(state_features, preference_vector, mode="policy")
            
            # Add exploration noise
            if exploration and self.training:
                noise = torch.normal(0, 0.1, size=action.shape).to(self.device)
                action = action + noise
                action = torch.clamp(action, 0, 1)
                action = action / (action.sum(dim=1, keepdim=True) + 1e-8)
            
            return action