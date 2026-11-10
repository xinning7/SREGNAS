


def is_cuda_oom(error):
    import torch

    cause = getattr(error, "cause", error)
    return isinstance(
        cause, torch.cuda.OutOfMemoryError
    ) or "CUDA out of memory" in str(cause)
