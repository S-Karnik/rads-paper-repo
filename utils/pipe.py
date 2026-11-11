from wah.diffusion.models import RealisticVision, StableDiffusion

__all__ = [
    "load_pipe",
]


def load_pipe(
    pipe_name: str,
    scheduler: str,
) -> StableDiffusion:
    if pipe_name == "SDV1":
        return StableDiffusion(
            version="1.4",
            scheduler=scheduler,
            variant="fp16",
            verbose=False,
            safety_check=False,
            torch_dtype=torch.float16
        )
    elif pipe_name == "RealVis":
        return RealisticVision(
            version="1.4",
            scheduler=scheduler,
            variant="fp16",
            verbose=False,
            safety_check=False,
        )
    else:
        raise ValueError(f"Unsupported pipe: {pipe_name}")
