import numpy as np
import random
import torch

from utils.stateful_randomness import temp_seed

def _capture_states():
    """Capture global RNG states for random, numpy, torch (CPU + CUDA)."""
    states = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return states


def _restore_states(states):
    """Restore global RNG states for random, numpy, torch (CPU + CUDA)."""
    random.setstate(states["py"])
    np.random.set_state(states["np"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def _sample_all():
    """
    Draw one sample from each RNG source.
    Returns plain Python types / numpy arrays / torch tensors.
    """
    py = random.random()
    npv = np.random.randn(4).astype(np.float64)

    cpu = torch.randn(4, dtype=torch.float32)

    if torch.cuda.is_available():
        gpu = torch.randn(4, device="cuda", dtype=torch.float32).cpu()
    else:
        gpu = None

    return py, npv, cpu, gpu


def _assert_samples_equal(a, b, *, atol=0.0, rtol=0.0):
    """Strict-ish equality across the mixed return types from _sample_all()."""
    a_py, a_np, a_cpu, a_gpu = a
    b_py, b_np, b_cpu, b_gpu = b

    assert a_py == b_py, f"python random mismatch: {a_py} vs {b_py}"
    assert np.array_equal(a_np, b_np), f"numpy mismatch:\n{a_np}\nvs\n{b_np}"
    assert torch.equal(a_cpu, b_cpu), f"torch CPU mismatch:\n{a_cpu}\nvs\n{b_cpu}"

    if (a_gpu is None) != (b_gpu is None):
        raise AssertionError("torch CUDA availability mismatch between samples")

    if a_gpu is not None:
        assert torch.equal(a_gpu, b_gpu), f"torch CUDA mismatch:\n{a_gpu}\nvs\n{b_gpu}"


def test_decorator_is_deterministic() -> None:
    # Under test.
    @temp_seed(123)
    def f():
        return _sample_all()

    out1 = f()
    out2 = f()

    # Postcondition.
    _assert_samples_equal(out1, out2)
    print("test_decorator_is_deterministic passed!")


def test_state_is_restored_after_call() -> None:
    # Precondition: snapshot current global RNG states
    s0 = _capture_states()

    # Compute the baseline "next samples" from this state
    baseline = _sample_all()

    # Reset back to s0
    _restore_states(s0)

    # Under test: call decorated function (should not perturb global state)
    @temp_seed(999)
    def f():
        # Consume randomness inside the decorated region
        _ = _sample_all()
        _ = _sample_all()

    f()

    # Postcondition: the next sample should match baseline
    after = _sample_all()
    _assert_samples_equal(baseline, after)
    print("test_state_is_restored_after_call passed!")


def test_state_is_restored_on_exception() -> None:
    # Precondition
    s0 = _capture_states()
    baseline = _sample_all()
    _restore_states(s0)

    # Under test
    @temp_seed(42)
    def f_raises():
        _ = _sample_all()
        raise RuntimeError("intentional error")

    try:
        f_raises()
    except RuntimeError:
        pass

    # Postcondition
    after = _sample_all()
    _assert_samples_equal(baseline, after)
    print("test_state_is_restored_on_exception passed!")


def test_different_seeds_change_output() -> None:
    # Under test.
    @temp_seed(1)
    def f1():
        return _sample_all()

    @temp_seed(2)
    def f2():
        return _sample_all()

    a = f1()
    b = f2()

    # Postcondition
    assert a[0] != b[0], f"expected different python random draws, got {a[0]} and {b[0]}"
    print("test_different_seeds_change_output passed!")


if __name__ == "__main__":
    test_decorator_is_deterministic()
    test_state_is_restored_after_call()
    test_state_is_restored_on_exception()
    test_different_seeds_change_output()
