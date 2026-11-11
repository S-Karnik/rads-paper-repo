import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import argparse
import itertools
import math
from tqdm import tqdm

import utils.dataset as dataset_utils

LOG_ALPHA_MIN = math.log(0.01)
LOG_ALPHA_MAX = math.log(0.5)

# Added bounds for Lagrange Multiplier (Lambda) to prevent instability
LOG_LAMBDA_MIN = math.log(0.01)
LOG_LAMBDA_MAX = math.log(1.0)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def run_evaluation(simulator, prompts, seeds, batch_size):
    tasks = list(itertools.product(prompts, seeds))
    
    avg_reward = 0.
    avg_l2_norm = 0.
    avg_clip_score = 0.
    total_episodes = 0
    
    for task_batch in tqdm(chunker(tasks, batch_size), desc="Eval Batches", leave=False, total=len(tasks)//batch_size if len(tasks) > 0 else 0):
        if not task_batch:
            continue
            
        batch_prompts = [t[0] for t in task_batch]
        batch_seeds = [t[1] for t in task_batch]
        
        batch_valid_prompts = []
        batch_valid_seeds = []

        for p, s in zip(batch_prompts, batch_seeds):
            try:
                simulator.get_target_images([p])
                batch_valid_prompts.append(p)
                batch_valid_seeds.append(s)
            except FileNotFoundError:
                continue
        
        if not batch_valid_prompts:
            continue

        eval_iterator = simulator.run_batch_episode(batch_valid_prompts, seeds=batch_valid_seeds)
        
        batch_rewards = [0.0] * len(batch_valid_prompts)
        
        for transition in eval_iterator:
            for i in range(len(batch_valid_prompts)):
                batch_rewards[i] += transition.task_reward[i]
                
                if transition.final_reward_components[i] is not None:
                    l2, text_s, img_s = transition.final_reward_components[i]
                    avg_l2_norm += l2
                    avg_clip_score += text_s 
        
        avg_reward += sum(batch_rewards)
        total_episodes += len(batch_valid_prompts)
        print(f"Current Avg Reward: {avg_reward / total_episodes}")

    metrics = {
        "avg_reward": avg_reward / total_episodes if total_episodes > 0 else 0.0,
        "transformed_l2_norm": avg_l2_norm / total_episodes if total_episodes > 0 else 0.0,
        "avg_clip_score": avg_clip_score / total_episodes if total_episodes > 0 else 0.0,
        "episodes": total_episodes
    }
    
    return metrics

def load_dataset(args):
    dataset_obj = dataset_utils.load_prompts(args.pipe)
    return dataset_obj.data

class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

        self._prompt_to_cap_id = {}
        self._cap_table = []

    def register_caps(self, prompts, caps_np):
        cap_ids = np.empty(len(prompts), dtype=np.int64)
        for i, p in enumerate(prompts):
            cid = self._prompt_to_cap_id.get(p, None)
            if cid is None:
                cid = len(self._cap_table)
                self._prompt_to_cap_id[p] = cid
                self._cap_table.append(caps_np[i].astype(np.float32, copy=False))
            cap_ids[i] = cid
        return cap_ids

    def _lookup_caps(self, cap_ids):
        return np.stack([self._cap_table[int(cid)] for cid in cap_ids], axis=0)

    # Renamed arguments for clarity: task_reward, safety_reward
    def push(self, state_img, state_cap_id, action, task_reward, safety_reward, next_state_img, next_state_cap_id, done, current_timestep):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            state_img,
            int(state_cap_id),
            action,
            task_reward,
            safety_reward,
            next_state_img,
            int(next_state_cap_id),
            done,
            current_timestep,
        )
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, state_imgs, state_cap_ids, actions, task_rewards, safety_rewards, next_state_imgs, next_state_cap_ids, dones, current_timesteps):
        batch_size = len(task_rewards)
        for i in range(batch_size):
            self.push(
                state_imgs[i],
                state_cap_ids[i],
                actions[i],
                task_rewards[i],
                safety_rewards[i],
                next_state_imgs[i],
                next_state_cap_ids[i],
                dones[i],
                current_timesteps[i],
            )

    def sample(self, batch_size):
        batch = self.rng.choice(len(self.buffer), batch_size)

        state_imgs, state_cap_ids, actions, task_rewards, safety_rewards, next_state_imgs, next_state_cap_ids, dones, current_timesteps = [], [], [], [], [], [], [], [], []

        for idx in batch:
            si, scid, a, tr, sr, nsi, nscid, d, ct = self.buffer[idx]
            state_imgs.append(si)
            state_cap_ids.append(scid)
            actions.append(a)
            task_rewards.append(tr)
            safety_rewards.append(sr)
            next_state_imgs.append(nsi)
            next_state_cap_ids.append(nscid)
            dones.append(d)
            current_timesteps.append(ct)

        state_caps_np = self._lookup_caps(state_cap_ids)
        next_state_caps_np = self._lookup_caps(next_state_cap_ids)

        return (
            torch.FloatTensor(np.stack(state_imgs)),
            torch.FloatTensor(state_caps_np),
            torch.FloatTensor(np.stack(actions)),
            torch.FloatTensor(np.stack(task_rewards)).unsqueeze(1),
            torch.FloatTensor(np.stack(safety_rewards)).unsqueeze(1),
            torch.FloatTensor(np.stack(next_state_imgs)),
            torch.FloatTensor(next_state_caps_np),
            torch.FloatTensor(np.stack(dones)).unsqueeze(1),
            torch.FloatTensor(np.stack(current_timesteps)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SharedEncoder(nn.Module):
    def __init__(self, caption_dim, time_embed_dim=16, encoded_dim=256, latent_resolution=64):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(8, 128)
        final_dim = latent_resolution // 8
        self.flat_img_size = 128 * final_dim * final_dim
        self.fc_img = nn.Linear(self.flat_img_size, 256)
        self.flat_cap_size = 2 * caption_dim 
        self.fc_cap = nn.Linear(self.flat_cap_size, 256)
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
        )
        self.fc_joint = nn.Linear(512 + time_embed_dim, encoded_dim)
        self.ln_joint = nn.LayerNorm(encoded_dim)
        self.apply(weights_init_)

    def forward(self, img, cap, timestep):
        x = F.relu(self.bn1(self.conv1(img)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x_img = F.relu(self.fc_img(x))
        if cap.dim() == 3: 
            x_c = cap.view(cap.size(0), -1) 
            x_cap = F.relu(self.fc_cap(x_c))
        else:
             raise ValueError(f"Expected 3D caption tensor [B, 2, 64], got {cap.shape}")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=x_img.device, dtype=torch.float32)
        timestep = timestep.to(dtype=torch.float32)

        if timestep.dim() == 0:
            timestep = timestep.reshape(1, 1).expand(x_img.size(0), 1)
        elif timestep.dim() == 1:
             if timestep.size(0) == 1:
                 timestep = timestep.expand(x_img.size(0)).unsqueeze(1)
             else:
                 timestep = timestep.unsqueeze(1)
        
        t_emb = self.time_mlp(timestep)
        x_joint = torch.cat([x_img, x_cap, t_emb], dim=1)
        return F.relu(self.ln_joint(self.fc_joint(x_joint)))

class Actor(nn.Module):
    def __init__(self, hidden_dim, action_dim=None, action_space=None):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)
        self.apply(weights_init_)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state_features):
        x = F.relu(self.ln1(self.linear1(state_features)))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-6, max=2)
        return mean, log_std

    def sample(self, state_features):
        mean, log_std = self.forward(state_features)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(hidden_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.linear4 = nn.Linear(hidden_dim + action_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)
        self.apply(weights_init_)

    def forward(self, state_features, action):
        xu = torch.cat([state_features, action], 1)
        x1 = F.relu(self.ln1(self.linear1(xu)))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.ln4(self.linear4(xu)))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

# --- SAFETY CRITIC (Q(s,a)) ---
# Similar architecture to Critic, but represents the Safety Q-Function
class SafetyCritic(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(SafetyCritic, self).__init__()
        self.linear1 = nn.Linear(hidden_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.apply(weights_init_)

    def forward(self, state_features, action):
        xu = torch.cat([state_features, action], 1)
        x1 = F.relu(self.ln1(self.linear1(xu)))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class SAC:
    def __init__(self, action_dim, caption_dim, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_reg = args.action_reg
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        
        self.safety_target = args.safety_target
        self.learn_lambda = args.learn_lambda

        # Determine latent resolution based on pipe
        latent_resolution = 96 if args.pipe == "SDV2" else 64
        self.encoder = SharedEncoder(caption_dim, time_embed_dim=args.time_embed_dim, latent_resolution=latent_resolution).to(self.device)
        self.actor = Actor(256, action_dim, None).to(self.device)
        
        # TASK Critic (Q)
        self.critic = Critic(256, action_dim).to(self.device)
        
        self.do_not_use_safety_critic = args.do_not_use_safety_critic
        self.sum_ablation = args.sum_ablation

        if not self.do_not_use_safety_critic:
            # SAFETY Critic (Q) - Restored Q(s,a)
            self.safety_critic = SafetyCritic(256, action_dim).to(self.device)
        
        if args.saved_sac_actor_path is not None:
            self.actor.load_state_dict(torch.load(args.saved_sac_actor_path))
        if args.saved_sac_critic_path is not None:
            self.critic.load_state_dict(torch.load(args.saved_sac_critic_path))
        if args.saved_sac_encoder_path is not None:
            self.encoder.load_state_dict(torch.load(args.saved_sac_encoder_path))
        
        if not self.do_not_use_safety_critic:
            if args.saved_sac_safety_critic_path is not None:
                self.safety_critic.load_state_dict(torch.load(args.saved_sac_safety_critic_path))
        
        self.critic_target = Critic(256, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        if not self.do_not_use_safety_critic:
            self.safety_critic_target = SafetyCritic(256, action_dim).to(self.device)
            self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
        
        self.encoder_target = SharedEncoder(caption_dim, time_embed_dim=args.time_embed_dim, latent_resolution=latent_resolution).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # Optimizers
        self.critic_optim = optim.Adam(list(self.critic.parameters()) + list(self.encoder.parameters()), lr=args.critic_lr)
        if not self.do_not_use_safety_critic:
            self.safety_optim = optim.Adam(self.safety_critic.parameters(), lr=args.critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        if not self.do_not_use_safety_critic:
            self.log_lambda = torch.tensor([np.log(0.5)], dtype=torch.float32, requires_grad=True, device=self.device)
            self.lambda_optim = optim.Adam([self.log_lambda], lr=args.lambda_lr) 

        self.automatic_entropy_tuning = True
        if self.automatic_entropy_tuning:
            self.entropy_dim = action_dim 
            self.entropy_scale = 1.0 
            self.target_entropy = -1.0
            self.log_alpha = torch.tensor([np.log(args.alpha)], dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.alpha_lr)

    def train(self):
        self.encoder.train(); self.actor.train(); self.critic.train()
        if not self.do_not_use_safety_critic:
            self.safety_critic.train()

    def eval(self):
        self.encoder.eval(); self.actor.eval(); self.critic.eval()
        if not self.do_not_use_safety_critic:
            self.safety_critic.eval()

    def select_action(self, state_img, state_cap, current_timestep, evaluate=False):
        if isinstance(state_img, np.ndarray):
            state_img = torch.FloatTensor(state_img)
            state_cap = torch.FloatTensor(state_cap)
        state_img = state_img.to(device=self.device, dtype=torch.float32)
        state_cap = state_cap.to(device=self.device, dtype=torch.float32)
        if state_img.dim() == 3: state_img = state_img.unsqueeze(0)
        if state_cap.dim() == 2: state_cap = state_cap.unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(state_img, state_cap, current_timestep)
            if evaluate:
                _, _, action = self.actor.sample(features)
            else:
                action, _, _ = self.actor.sample(features)
        return action.detach()

    def update_parameters(self, memory, batch_size, updates):
        # Unpack
        state_img_batch, state_cap_batch, action_batch, task_reward_batch, safety_reward_batch, next_state_img_batch, next_state_cap_batch, mask_batch, current_timestep_batch = memory.sample(batch_size)

        state_img_batch = state_img_batch.to(self.device)
        state_cap_batch = state_cap_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        task_reward_batch = task_reward_batch.to(self.device)
        safety_reward_batch = safety_reward_batch.to(self.device)
        next_state_img_batch = next_state_img_batch.to(self.device)
        next_state_cap_batch = next_state_cap_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)
        current_timestep_batch = current_timestep_batch.to(self.device)

        # --- 1. Task Critic (Q) Update - Standard Bellman ---
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_features = self.encoder_target(next_state_img_batch, next_state_cap_batch, current_timestep_batch)
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_features)
            qf1_next_target, qf2_next_target = self.critic_target(next_features, next_state_action)
            next_log_pi_scaled = (next_state_log_pi / self.entropy_dim) * self.entropy_scale
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi_scaled
            
            # Bellman Equation for Task
            next_q_value = task_reward_batch + self.gamma * mask_batch * min_qf_next_target
        
        features = self.encoder(state_img_batch, state_cap_batch, current_timestep_batch)
        qf1, qf2 = self.critic(features, action_batch)  
        qf_loss = F.smooth_l1_loss(qf1, next_q_value) + F.smooth_l1_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        qf_loss.backward(retain_graph=True) 
        
        # --- 2. Safety Critic (Q) Update ---
        safety_loss = 0.0
        if not self.do_not_use_safety_critic:
            with torch.no_grad():
                # For Safety Q target, we need Q(next_state, next_action)
                sf_next = self.safety_critic_target(next_features, next_state_action)
                
                # Reachability Formula adapted for Q:
                # y = (1-gamma)R + gamma * min(R, Q(s',a'))
                # If done (mask=0), target is just R.
                
                # term2: Future worst-case
                min_future = torch.min(safety_reward_batch, sf_next)
                
                target_safety_q = (1 - self.gamma) * safety_reward_batch + \
                                  self.gamma * (mask_batch * min_future + (1 - mask_batch) * safety_reward_batch)

            sf = self.safety_critic(features, action_batch)
            loss_safety = F.mse_loss(sf, target_safety_q)
            safety_loss = loss_safety.item()
            
            self.safety_optim.zero_grad()
            loss_safety.backward()
            torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), max_norm=1.0)
            self.safety_optim.step()
        elif self.sum_ablation:
            with torch.no_grad():
                # For Safety Q target, we need Q(next_state, next_action)
                sf_next = self.safety_critic_target(next_features, next_state_action)
                
                # Sum Q Target (Traditional Bellman Formulation)              
                target_safety_q = safety_reward_batch + self.gamma * mask_batch * sf_next

            sf = self.safety_critic(features, action_batch)
            loss_safety = F.mse_loss(sf, target_safety_q)
            safety_loss = loss_safety.item()
            
            self.safety_optim.zero_grad()
            loss_safety.backward()
            torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), max_norm=1.0)
            self.safety_optim.step()
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # --- 3. Lambda Update ---
        lambda_val = 0.0
        if not self.do_not_use_safety_critic:
            # We want Q_safe >= Target (0). 
            # Loss = lambda * (Q_safe - Target)
            # Update: lambda += lr * (Target - Q_safe)
            with torch.no_grad():
                # Use current batch safety prediction average
                current_safety_mean = sf.mean()
                
            lambda_val = self.log_lambda.exp()
            lambda_loss = lambda_val * (current_safety_mean - self.safety_target)
            
            self.lambda_optim.zero_grad()
            lambda_loss.backward()
            self.lambda_optim.step()
            
            with torch.no_grad():
                self.log_lambda.clamp_(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX)
            lambda_val = lambda_val.item()

        # --- 4. Actor Update ---
        features_detached = features.detach()
        pi, log_pi, mean_action = self.actor.sample(features_detached)
        
        qf1_pi, qf2_pi = self.critic(features_detached, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi) 
        
        log_pi_scaled = (log_pi / self.entropy_dim) * self.entropy_scale
        
        # Objective: Maximize Q + Lambda * Q_safe
        if not self.do_not_use_safety_critic:
            # Query Safety Critic (Q(s, pi)) - gradients flow through pi!
            sf_pi = self.safety_critic(features_detached, pi)
            lambda_detached = self.log_lambda.exp().detach()
            actor_objective = min_qf_pi + (lambda_detached * sf_pi) - (alpha * log_pi_scaled)
        else:
            actor_objective = min_qf_pi - (alpha * log_pi_scaled)
        
        policy_loss = -actor_objective.mean()

        pi_sq_mean = (pi ** 2).mean()
        
        if self.action_reg > 0:
            policy_loss += self.action_reg * pi_sq_mean

        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # --- 5. Alpha Update ---
        if self.automatic_entropy_tuning:
            log_pi_scaled = (log_pi / self.entropy_dim) * self.entropy_scale
            alpha_loss = -(self.log_alpha * (log_pi_scaled + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            with torch.no_grad():
                self.log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha
        else:
            alpha_loss = torch.tensor(0., device=self.device)
            alpha_tlogs = float(self.alpha)

        # Soft Updates
        if updates % 1 == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            # Update Safety Target
            if not self.do_not_use_safety_critic:
                for target_param, param in zip(self.safety_critic_target.parameters(), self.safety_critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf_loss.item(), safety_loss, policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), lambda_val, pi_sq_mean.item()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--actor_lr', type=float, default=0.0001)
    parser.add_argument('--critic_lr', type=float, default=0.0001)
    parser.add_argument('--alpha_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_lr', type=float, default=0.00003)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--caption_dim', type=int, default=64)
    parser.add_argument('--replay_buffer_capacity', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--saved_sac_model_path', type=str, default=None)
    parser.add_argument('--diffusion_model_type', type=str, default='sdv1')
    parser.add_argument('--num_timesteps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--pipe', type=str, default='SDV1')
    parser.add_argument('--scheduler', type=str, default='DDIM')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--start_steps', type=int, default=500)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--eval_seed_min', type=int, default=0)
    parser.add_argument('--eval_seed_max', type=int, default=2)
    parser.add_argument('--action_reg', type=float, default=1.0)
    parser.add_argument('--sim_batch_size', type=int, default=16)
    parser.add_argument('--time_embed_dim', type=int, default=16)
    parser.add_argument('--do_not_use_safety_critic', action='store_true', help="Disable safety critic training and usage")

    # Safety Arguments
    parser.add_argument('--safety_target', type=float, default=0.0, help="Target CLIP Score to maintain")
    parser.add_argument('--learn_lambda', type=bool, default=True, help="Whether to learn the Lagrangian multiplier")

    # Model paths
    parser.add_argument('--vae_path', type=str, default="clip_vae_final.pth")
    parser.add_argument('--saved_sac_actor_path', type=str, default=None)
    parser.add_argument('--saved_sac_critic_path', type=str, default=None)
    parser.add_argument('--saved_sac_encoder_path', type=str, default=None)
    parser.add_argument('--saved_sac_safety_critic_path', type=str, default=None)
    parser.add_argument('--tanh_l2_threshold', type=float, required=True, help="Threshold for Tanh L2 transformation")

    parser.add_argument('--sum_ablation', action='store_true', default=False, help="Use sum ablation")

    return parser

def parse_args():
    parser = get_parser()
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent = SAC(64, args.caption_dim, args)
    print("SAC initialized for Constrained Optimization (Max Structure s.t. CLIP >= Target)")