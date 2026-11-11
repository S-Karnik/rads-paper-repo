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
from train import sac_steer_lib
from utils import misc
from utils import model_types

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def train(args):
    # Initialize WandB
    wandb.init(project="diffusion-steering", config=args)

    # Global Seeding for Train Mode
    misc.set_seed(args.seed)

    # Load SAC Agent
    sac_agent = sac_steer_lib.SAC(
        action_dim=args.caption_dim,
        caption_dim=args.caption_dim,
        args=args,
    )
    
    # Load Checkpoint if provided
    if args.saved_sac_model_path:
        print(f"Loading model from {args.saved_sac_model_path}")

    os.makedirs("checkpoints", exist_ok=True)

    # Load Replay Buffer
    memory = sac_steer_lib.ReplayBuffer(args.replay_buffer_capacity, args.seed)

    # Load Dataset
    dataset = sac_steer_lib.load_dataset(args)

    # Load Simulator for Training
    train_simulator = steering_sim.SteeringSimulator(
        pipe_name=args.pipe,
        scheduler_name=args.scheduler,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        sac_agent=sac_agent,
        diffusion_model_type=model_types.ModelType.from_str(args.diffusion_model_type),
        vae_path=args.vae_path,
        tanh_l2_threshold=args.tanh_l2_threshold,
        device=sac_agent.device,
        train_mode=True,
    )

    # Load Simulator for Evaluation
    eval_simulator = steering_sim.SteeringSimulator(
        pipe_name=args.pipe,
        scheduler_name=args.scheduler,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        sac_agent=sac_agent,
        diffusion_model_type=model_types.ModelType.from_str(args.diffusion_model_type),
        vae_path=args.vae_path,
        tanh_l2_threshold=args.tanh_l2_threshold,
        device=sac_agent.device,
        train_mode=False,
    )

    total_numsteps = 0
    updates = 0

    # BATCH SIZE CONFIG
    PROMPT_BATCH_SIZE = args.sim_batch_size 

    if args.do_not_use_safety_critic:
        dir_name = f"checkpoints_{args.pipe}_{args.scheduler}_seed_{args.seed}_no_safety_critic"
    else:
        dir_name = f"checkpoints_{args.pipe}_{args.scheduler}_seed_{args.seed}"
    os.makedirs(dir_name, exist_ok=True)

    best_sum_score = -np.inf

    # Run Training
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        sac_agent.train()
        prompts = dataset['caption'].tolist() if hasattr(dataset['caption'], 'tolist') else list(dataset['caption'])
        print(f"Number of prompts: {len(prompts)}", flush=True)
        original_prompts = prompts.copy()
        random.shuffle(prompts)
        
        for prompt_batch in tqdm(chunker(prompts, PROMPT_BATCH_SIZE), desc="Training Batches", leave=False, total=len(prompts)//PROMPT_BATCH_SIZE):
            
            remaining_prompts = []
            for prompt in prompt_batch:
                try:
                    train_simulator.get_target_images([prompt])
                    remaining_prompts.append(prompt)
                except FileNotFoundError as e:
                    pass

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
                        q1_loss, safety_loss, pi_loss, alpha_loss, alpha_val, lambda_val, pi_sq_mean = sac_agent.update_parameters(memory, args.batch_size, updates)
                        updates += 1
                        
                        wandb.log({
                            "train/q_loss": q1_loss,
                            "train/safety_loss": safety_loss,
                            "train/policy_loss": pi_loss,
                            "train/alpha_loss": alpha_loss,
                            "train/alpha": alpha_val,
                            "train/lambda": lambda_val,
                            "train/pi_sq_mean": pi_sq_mean,
                            "train/steps": total_numsteps
                        })
                
        # Evaluation Loop
        if epoch % args.eval_interval == 0:
            eval_simulator.eval()
            eval_seeds = list(range(args.eval_seed_min, args.eval_seed_max + 1))
            
            metrics = sac_steer_lib.run_evaluation(
                simulator=eval_simulator,
                prompts=original_prompts,
                seeds=eval_seeds,
                batch_size=PROMPT_BATCH_SIZE
            )
            
            avg_reward = metrics['avg_reward']
            avg_clip_score = metrics['avg_clip_score']

            if args.do_not_use_safety_critic:
                sum_score = avg_clip_score
            else:
                sum_score = avg_clip_score + metrics['transformed_l2_norm']
                
            if best_sum_score < sum_score:
                best_sum_score = sum_score
                torch.save(sac_agent.actor.state_dict(), f"{dir_name}/sac_actor_best.pth")
                torch.save(sac_agent.critic.state_dict(), f"{dir_name}/sac_critic_best.pth")
                if not args.do_not_use_safety_critic:
                    torch.save(sac_agent.safety_critic.state_dict(), f"{dir_name}/sac_safety_critic_best.pth")
                torch.save(sac_agent.encoder.state_dict(), f"{dir_name}/sac_encoder_best.pth")
                print(f"Saved latest best model to {dir_name}/sac_actor_best.pth", flush=True)
            
            wandb.log({"eval/avg_reward": avg_reward, "eval/avg_clip_score": avg_clip_score, "epoch": epoch, "eval/transformed_l2_norm": metrics['transformed_l2_norm']})
            print(f"Epoch {epoch}: Eval Avg Reward: {avg_reward:.4f}, Avg Clip Score: {avg_clip_score:.4f}, Avg Transformed L2 Norm: {metrics['transformed_l2_norm']:.4f}")

            # Save Model
            torch.save(sac_agent.actor.state_dict(), f"{dir_name}/sac_actor_{epoch}.pth")
            torch.save(sac_agent.critic.state_dict(), f"{dir_name}/sac_critic_{epoch}.pth")
            if not args.do_not_use_safety_critic:
                torch.save(sac_agent.safety_critic.state_dict(), f"{dir_name}/sac_safety_critic_{epoch}.pth")
            torch.save(sac_agent.encoder.state_dict(), f"{dir_name}/sac_encoder_{epoch}.pth")
        
            print(f"Saved model to {dir_name}/sac_actor_{epoch}.pth", flush=True)

if __name__ == "__main__":
    args = sac_steer_lib.parse_args()
    train(args)