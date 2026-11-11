import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import itertools
from tqdm import tqdm

import utils.dataset as dataset_utils

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

    def sample_all(self):
        batch = np.arange(len(self.buffer))

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
        self.dropout = nn.Dropout2d(0.4)
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
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
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

class SafetyValue(nn.Module):
    def __init__(self, hidden_dim):
        super(SafetyValue, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.apply(weights_init_)

    def forward(self, state_features):
        x1 = F.relu(self.ln1(self.linear1(state_features)))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class VAgent:
    def __init__(self, action_dim, caption_dim, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.sum_ablation = args.sum_ablation
        
        latent_resolution = 96 if args.pipe == "SDV2" else 64
        self.encoder = SharedEncoder(caption_dim, time_embed_dim=args.time_embed_dim, latent_resolution=latent_resolution).to(self.device)
        self.safety_value = SafetyValue(256).to(self.device)
        
        if args.saved_sac_encoder_path is not None:
            self.encoder.load_state_dict(torch.load(args.saved_sac_encoder_path))
        if args.saved_sac_safety_critic_path is not None:
            self.safety_value.load_state_dict(torch.load(args.saved_sac_safety_critic_path))
        
        self.encoder_target = SharedEncoder(caption_dim, time_embed_dim=args.time_embed_dim, latent_resolution=latent_resolution).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.safety_value_target = SafetyValue(256).to(self.device)
        self.safety_value_target.load_state_dict(self.safety_value.state_dict())
        
        self.value_optim = optim.Adam(list(self.safety_value.parameters()) + list(self.encoder.parameters()), lr=args.critic_lr, weight_decay=1e-3)

    def train(self):
        self.encoder.train(); self.safety_value.train()

    def eval(self):
        self.encoder.eval(); self.safety_value.eval()

    def select_action(self, state_img, state_cap, current_timestep, evaluate=False):
        batch_size = state_img.shape[0] if isinstance(state_img, torch.Tensor) else 1
        if isinstance(state_img, np.ndarray):
            batch_size = state_img.shape[0]
        # V function doesn't need to select an action, we just return a zero vector
        action = torch.zeros(batch_size, self.action_dim, device=self.device)
        return action

    def update_parameters(self, memory, batch_size, updates):
        state_img_batch, state_cap_batch, _, _, safety_reward_batch, next_state_img_batch, next_state_cap_batch, mask_batch, current_timestep_batch = memory.sample(batch_size)

        state_img_batch = state_img_batch.to(self.device)
        state_cap_batch = state_cap_batch.to(self.device)
        safety_reward_batch = safety_reward_batch.to(self.device)
        next_state_img_batch = next_state_img_batch.to(self.device)
        next_state_cap_batch = next_state_cap_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)
        current_timestep_batch = current_timestep_batch.to(self.device)

        if self.encoder.training:
            state_img_batch = state_img_batch + torch.randn_like(state_img_batch) * 0.05

        features = self.encoder(state_img_batch, state_cap_batch, current_timestep_batch)
        vf = self.safety_value(features)

        with torch.no_grad():
            next_features = self.encoder_target(next_state_img_batch, next_state_cap_batch, current_timestep_batch)
            vf_next = self.safety_value_target(next_features)
            
            if self.sum_ablation:
                target_safety_v = safety_reward_batch + self.gamma * mask_batch * vf_next
            else:
                min_future = torch.min(safety_reward_batch, vf_next)
                target_safety_v = (1 - self.gamma) * safety_reward_batch + \
                                  self.gamma * (mask_batch * min_future + (1 - mask_batch) * safety_reward_batch)

        loss_value = F.mse_loss(vf, target_safety_v)
        
        self.value_optim.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_value.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.value_optim.step()

        if updates % 1 == 0:
            for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.safety_value_target.parameters(), self.safety_value.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return loss_value.item()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--critic_lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--caption_dim', type=int, default=64)
    parser.add_argument('--replay_buffer_capacity', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
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
    parser.add_argument('--sim_batch_size', type=int, default=16)
    parser.add_argument('--time_embed_dim', type=int, default=16)

    # Model paths
    parser.add_argument('--vae_path', type=str, default="clip_vae_final.pth")
    parser.add_argument('--saved_sac_encoder_path', type=str, default=None)
    parser.add_argument('--saved_sac_safety_critic_path', type=str, default=None)
    parser.add_argument('--tanh_l2_threshold', type=float, required=True, help="Threshold for Tanh L2 transformation")

    # Ablation Arguments
    parser.add_argument('--sum_ablation', action='store_true', default=False, help="Use sum ablation")
    
    parser.add_argument("--prompts_root", type=str, help="Root directory for prompts")

    return parser

def parse_args():
    parser = get_parser()
    return parser.parse_args()
