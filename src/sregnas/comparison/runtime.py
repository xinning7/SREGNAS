

import os


def configure_native_client():
    from sregnas.paths import require_gpu_permission

    require_gpu_permission()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = set(os.environ.get("SREGNAS_APPROVED_GPUS", "").split(","))
    if not visible or not set(visible.split(",")) <= allowed:
        raise RuntimeError(
            "native process requested a GPU outside the launch allocation"
        )


def request_handoff():
    raise RuntimeError(
        "legacy campaign handoff is not supported by the standalone launcher"
    )


def native_cell_start(dataset, phase):
    from sregnas.execution.startup import assert_cell_start

    flags = dict(sampler_impl="gpu", feature_residency="gpu", inference_residency="gpu")
    return assert_cell_start(
        dict(
            dataset=dataset.lower(),
            mode="native_search",
            phase=phase,
            **flags,
            expected_executor=flags,
        )
    )
