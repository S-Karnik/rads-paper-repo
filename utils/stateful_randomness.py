import random
from contextlib import ContextDecorator

class temp_seed(ContextDecorator):
    def __init__(
        self,
        seed: int,
        *,
        seed_random: bool = True,
        seed_numpy: bool = True,
        seed_torch: bool = True,
        seed_cuda: bool = True,
    ):
        self.seed = int(seed)
        self.seed_random = seed_random
        self.seed_numpy = seed_numpy
        self.seed_torch = seed_torch
        self.seed_cuda = seed_cuda

        self._py_state = None
        self._np_state = None
        self._torch_cpu_state = None
        self._torch_cuda_state = None

    def __enter__(self):
        # Save states
        if self.seed_random:
            self._py_state = random.getstate()

        if self.seed_numpy:
            import numpy as np
            self._np_state = np.random.get_state()

        if self.seed_torch:
            import torch
            self._torch_cpu_state = torch.get_rng_state()
            if self.seed_cuda and torch.cuda.is_available():
                self._torch_cuda_state = torch.cuda.get_rng_state_all()

        # Set seeds
        if self.seed_random:
            random.seed(self.seed)

        if self.seed_numpy:
            import numpy as np
            np.random.seed(self.seed)

        if self.seed_torch:
            import torch
            torch.manual_seed(self.seed)
            if self.seed_cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore states
        if self.seed_random and self._py_state is not None:
            random.setstate(self._py_state)

        if self.seed_numpy and self._np_state is not None:
            import numpy as np
            np.random.set_state(self._np_state)

        if self.seed_torch and self._torch_cpu_state is not None:
            import torch
            torch.set_rng_state(self._torch_cpu_state)
            if (
                self.seed_cuda
                and self._torch_cuda_state is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(self._torch_cuda_state)

        return False  # don't suppress exceptions
