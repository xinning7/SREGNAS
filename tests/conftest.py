import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("SREGNAS_ALLOW_GPU", None)
os.environ.pop("SREGNAS_EXECUTOR_OPTIONS", None)

import pytest
import torch

torch.set_num_threads(int(os.environ.get("SREGNAS_CPU_THREADS", "2")))


@pytest.fixture(autouse=True)
def prohibit_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU regression suite must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
