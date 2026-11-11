import diffusers
import numpy as np
import os
import random
import torch
from tqdm import tqdm
import transformers
import wandb

# Suppress verbose output
diffusers.utils.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

from diffusion_steering_simulator import steering_sim
from train import v_steer_lib
from utils import misc
from utils import model_types

from PIL import Image

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def load_train_and_val_prompts(prompts_root, diffusion_model_name, prompts, df):
    train_prompts = []
    val_prompts = []

    for prompt in prompts:
        try:
            caption_element_idx = df[df['caption'] == prompt].index[0]
            image_path = os.path.join(
                f"{prompts_root}/{diffusion_model_name}",
                "image", f"{caption_element_idx}.png",
            )
            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((512, 512), Image.Resampling.BICUBIC)
            train_prompts.append(prompt)
        except Exception as e:            
            val_prompts.append(prompt)
    return train_prompts, val_prompts

def train(args):
    # Initialize WandB
    wandb.init(project="diffusion-value-guidance", config=args)

    # Global Seeding for Train Mode
    misc.set_seed(args.seed)

    # Load VAgent
    v_agent = v_steer_lib.VAgent(
        action_dim=args.caption_dim,
        caption_dim=args.caption_dim,
        args=args,
    )

    # Load Replay Buffer
    memory = v_steer_lib.ReplayBuffer(args.replay_buffer_capacity, args.seed)

    # Load Dataset
    dataset = v_steer_lib.load_dataset(args)
    all_prompts = dataset['caption'].tolist() if hasattr(dataset['caption'], 'tolist') else list(dataset['caption'])
    train_prompts, val_prompts = load_train_and_val_prompts(args.prompts_root, args.pipe, all_prompts, dataset)
    print(f"Loaded {len(train_prompts)} train prompts and {len(val_prompts)} val prompts", flush=True)

    # Load Simulator for Training
    train_simulator = steering_sim.SteeringSimulator(
        pipe_name=args.pipe,
        scheduler_name=args.scheduler,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        sac_agent=v_agent,
        diffusion_model_type=model_types.ModelType.from_str(args.diffusion_model_type),
        vae_path=args.vae_path,
        tanh_l2_threshold=args.tanh_l2_threshold,
        device=v_agent.device,
        train_mode=True,
    )

    total_numsteps = 0
    updates = 0

    # BATCH SIZE CONFIG
    PROMPT_BATCH_SIZE = args.sim_batch_size 

    if args.use_residual:
        dir_name = f"checkpoints_safety_value_function_only_{args.pipe}_residual_{args.scheduler}_seed_{args.seed}"
    else:
        dir_name = f"checkpoints_safety_value_function_only_{args.pipe}_{args.scheduler}_seed_{args.seed}"
    os.makedirs(dir_name, exist_ok=True)

    # Run Training
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        v_agent.train()
        prompts = train_prompts.copy()
        random.shuffle(prompts)
        
        for prompt_batch in tqdm(chunker(prompts, PROMPT_BATCH_SIZE), desc="Training Batches", leave=False, total=len(prompts)//PROMPT_BATCH_SIZE):
            
            # Since we pre-filtered train_prompts, we can just use prompt_batch
            remaining_prompts = prompt_batch

            if len(remaining_prompts) == 0:
                continue
            
            # Run Batch Episode
            iterator = train_simulator.run_batch_episode(remaining_prompts)
            
            for transition in iterator:                
                batch_len = len(transition.task_reward)
                
                s_img = transition.state_img.cpu().numpy()     # [B, 4, 64, 64]
                act   = transition.action.cpu().numpy()        # [B, 64]
                ns_img = transition.next_state_img.cpu().numpy()

                caps_np = transition.state_cap.cpu().numpy()   # [B, 2, 64]
                cap_ids = memory.register_caps(remaining_prompts, caps_np)
                next_cap_ids = cap_ids

                mask = 1 if not transition.done else 0
                masks = [mask] * batch_len

                memory.push_batch(
                    s_img,
                    cap_ids,
                    act,
                    transition.task_reward,   # Sparse CLIP
                    transition.safety_reward, # Dense Negative L2
                    ns_img,
                    next_cap_ids,
                    masks,
                    transition.timestep.cpu().numpy() if isinstance(transition.timestep, torch.Tensor) else transition.timestep
                )
                
                total_numsteps += batch_len

                if len(memory) > args.start_steps:
                    for _ in range(args.updates_per_step):
                        v_loss = v_agent.update_parameters(memory, args.batch_size, updates)
                        updates += 1
                        
                        wandb.log({
                            "train/safety_v_loss": v_loss,
                            "train/steps": total_numsteps
                        })
                
        # Save Model
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            torch.save(v_agent.safety_value.state_dict(), f"{dir_name}/v_value_{epoch}.pth")
            torch.save(v_agent.encoder.state_dict(), f"{dir_name}/v_encoder_{epoch}.pth")
        
            print(f"Saved model to {dir_name}/v_value_{epoch}.pth", flush=True)

        if len(val_prompts) > 0:
            print(f"\\n--- Running Validation on Val Prompts at step {epoch} ---")
            
            # To get validation loss without stepping the optimizer, we evaluate TD-error.
            v_agent.eval()
            
            # Run simulator on val_prompts in eval mode to collect transitions
            # We need a new simulator for validation since train_simulator is in train_mode
            eval_simulator = steering_sim.SteeringSimulator(
                pipe_name=args.pipe, scheduler_name=args.scheduler, num_timesteps=args.num_timesteps,
                guidance_scale=args.guidance_scale, sac_agent=v_agent,
                diffusion_model_type=model_types.ModelType.from_str(args.diffusion_model_type),
                vae_path=args.vae_path, tanh_l2_threshold=args.tanh_l2_threshold,
                device=v_agent.device, train_mode=False,
            )
            
            val_memory = v_steer_lib.ReplayBuffer(args.replay_buffer_capacity, args.seed)
            val_iterator = eval_simulator.run_batch_episode(val_prompts[:args.sim_batch_size], seeds=[42]*min(len(val_prompts), args.sim_batch_size))
            
            for transition in val_iterator:                
                batch_len = len(transition.task_reward)
                s_img = transition.state_img.cpu().numpy()
                act   = transition.action.cpu().numpy()
                ns_img = transition.next_state_img.cpu().numpy()
                caps_np = transition.state_cap.cpu().numpy()
                cap_ids = val_memory.register_caps(val_prompts[:args.sim_batch_size], caps_np)
                masks = [1 if not transition.done else 0] * batch_len

                val_memory.push_batch(
                    s_img, cap_ids, act, transition.task_reward, transition.safety_reward,
                    ns_img, cap_ids, masks, transition.timestep.cpu().numpy() if isinstance(transition.timestep, torch.Tensor) else transition.timestep
                )
            
            # Compute Val Loss over the ENTIRE validation buffer
            import torch.nn.functional as F
            with torch.no_grad():
                state_img_all, state_cap_all, action_all, task_reward_all, safety_reward_all, next_state_img_all, next_state_cap_all, mask_all, current_timestep_all = val_memory.sample_all()
                
                total_val_loss = 0.0
                num_val_items = len(state_img_all)
                chunk_size = args.batch_size
                
                for i in range(0, num_val_items, chunk_size):
                    end_i = min(i + chunk_size, num_val_items)
                    
                    state_img_batch = state_img_all[i:end_i].to(v_agent.device)
                    state_cap_batch = state_cap_all[i:end_i].to(v_agent.device)
                    safety_reward_batch = safety_reward_all[i:end_i].to(v_agent.device)
                    next_state_img_batch = next_state_img_all[i:end_i].to(v_agent.device)
                    next_state_cap_batch = next_state_cap_all[i:end_i].to(v_agent.device)
                    mask_batch = mask_all[i:end_i].to(v_agent.device)
                    current_timestep_batch = current_timestep_all[i:end_i].to(v_agent.device)
                    
                    features = v_agent.encoder(state_img_batch, state_cap_batch, current_timestep_batch)
                    vf = v_agent.safety_value(features)
                    
                    next_features = v_agent.encoder_target(next_state_img_batch, next_state_cap_batch, current_timestep_batch)
                    vf_next = v_agent.safety_value_target(next_features)
                    
                    min_future = torch.min(safety_reward_batch, vf_next)
                    target_safety_v = (1 - v_agent.gamma) * safety_reward_batch + \
                                        v_agent.gamma * (mask_batch * min_future + (1 - mask_batch) * safety_reward_batch)
                                        
                    chunk_loss = F.mse_loss(vf, target_safety_v, reduction='sum')
                    total_val_loss += chunk_loss.item()
                    
                val_loss = total_val_loss / num_val_items
            
            print(f"Validation V-Loss (Full Buffer): {val_loss:.6f}")
            wandb.log({
                "val/safety_v_loss": val_loss,
                "epoch": epoch
            })

if __name__ == "__main__":
    args = v_steer_lib.parse_args()
    train(args)