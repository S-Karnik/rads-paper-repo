<div align="center">

# Steering Away from Memorization: Reachability-Constrained Reinforcement Learning for Text-to-Image Diffusion

**Sathwik Karnik**<sup>1,&#42;</sup> &nbsp;&nbsp; **Juyeop Kim**<sup>2,&#42;</sup> &nbsp;&nbsp; **Sanmi Koyejo**<sup>1,&dagger;</sup> &nbsp;&nbsp; **Jong-Seok Lee**<sup>2,&dagger;</sup> &nbsp;&nbsp; **Somil Bansal**<sup>1,&dagger;</sup>

<sup>1</sup>Stanford University, United States<br>
<sup>2</sup>School of Integrated Technology & BK21 Graduate Program in Intelligent Semiconductor Technology, Yonsei University, Korea

`{sathwik,sanmi,somil}@stanford.edu` &nbsp;&nbsp; `{juyeopkim,jong-seok.lee}@yonsei.ac.kr`

**NeurIPS 2026**

🌐 [Project Page](https://s-karnik.github.io/rads-memorization-project-page/) | 📄 [Paper](https://arxiv.org/abs/2603.00140)

</div>

This repository contains the official code for Reachability-Aware Diffusion Steering.

## Setup

1. Start with a fresh conda environment:
```bash
conda create -n diffusion_steering python=3.8.20
conda activate diffusion_steering
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Datasets

1. Download Webster prompts/images:
```bash
python download_prompts.py
```
2. Download MemBench prompts/images:
```bash
python download_membench_images.py
```

## Training

### Training VAE

To train the VAE, run:
```bash
python clip_autoencoder/train_autoencoder.py --validation_parquet_path PATH_TO_VALIDATION_PARQUET --diffusion_model_name DIFFUSION_MODEL_NAME --prompts_root PATH_TO_PROMPTS_ROOT
```
Here, `PATH_TO_VALIDATION_PARQUET` is the path to the validation parquet file (from the download_prompts.py script), `DIFFUSION_MODEL_NAME` is the name of the diffusion model (SDV1 or RealVis), and `PATH_TO_PROMPTS_ROOT` is the root directory of the prompts (from the download_prompts.py script).

### Training the Constrained Soft Actor Critic

To train the Constrained Soft Actor Critic, run:
```bash
python3 train/train_sac.py --vae_path clip_vae/step\=45000.pth --sim_batch_size 32 --eval_interval 5 --updates_per_step 1 --critic_lr 0.00003 --alpha_lr 0.00003 --actor_lr 0.0001 --action_reg 0.0 --tanh_l2_threshold 9 --num_epochs 90 --seed 42 --scheduler DDIM --pipe SDV1 --diffusion_model_type sdv1 --lambda_lr 0.00003
```

You can choose to modify diffusion_model_type to realvis if you want to train on RealVis. You can also modify the scheduler to DDPM.

For SD v2.1, change the arguments to the following:

```bash
python3 train/train_sac.py --vae_path clip_vae_sdv2/step\=45000.pth --sim_batch_size 16 --eval_interval 5 --updates_per_step 1 --critic_lr 0.00003 --alpha_lr 0.00001 --actor_lr 0.0001 --action_reg 0.0 --tanh_l2_threshold 11 --num_epochs 90 --seed 42 --scheduler DDIM --pipe SDV2 --diffusion_model_type sdv2 --lambda_lr 0.0001
```

### Evaluation

To evaluate the Constrained Soft Actor Critic, run:
```bash
bash generate.sh
```

Then to compute the metrics, run each of the following scripts:

For SSCD_target:
```bash
python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --train-seed 41
python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --train-seed 42
python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --train-seed 43
python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --train-seed 44
python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --train-seed 45

python3 sscd_score_dataset.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --reference-dir prompts/SDV1/image/ --aggregate-metrics
```

For SSCD_seeds:
```bash 
python3 sscd_score_seed_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM

python3 sscd_score_seed_agg.py --pipe SDV1 --scheduler DDIM
```

For SSCD_prompt:
```bash 
python3 sscd_score_prompt_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 41
python3 sscd_score_prompt_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 42
python3 sscd_score_prompt_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 43
python3 sscd_score_prompt_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 44
python3 sscd_score_prompt_compute.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 45

python3 sscd_score_prompt_agg.py --pipe SDV1 --scheduler DDIM
```

For FID:
```bash
python3 fid_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 41
python3 fid_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 42
python3 fid_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 43
python3 fid_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 44
python3 fid_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 45

python3 fid_score.py --pipe SDV1 --scheduler DDIM --aggregate-metrics
```

For CLIP:
```bash
python3 clip_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 41
python3 clip_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 42
python3 clip_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 43
python3 clip_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 44
python3 clip_score.py --mitigate sac_safety_critic --pipe SDV1 --scheduler DDIM --train-seed 45

python3 clip_score.py --pipe SDV1 --scheduler DDIM --aggregate-metrics
```

For any of these, you should be able to run different mitigations by changing the `--mitigate` flag. Possible options are:
`none`, `wen_iclr2024`, `ren_eccv2024`, `hintersdorf_nips2024`, `jain_cvpr2025`, `sac_safety_critic`.

Additionally, you can change the pipe by changing the `--pipe` flag. Possible options are:
`SDV1`, `RealVis`, `SDV2`.

Lastly, you can change the scheduler by changing the `--scheduler` flag. Possible options are:
`DDIM`, `DDPM`.

### Plotting

To regenerate the plot in Figures 5 and 6, run:
```bash
python3 pareto_frontier.py
```
