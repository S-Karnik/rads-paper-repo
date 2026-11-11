from enum import Enum

class ModelType(Enum):
    STABLE_DIFFUSION_V1 = "sdv1"
    STABLE_DIFFUSION_V2 = "sdv2"
    REAL_VISION = "realvis"
    
    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "ModelType":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid model type: {value}")

