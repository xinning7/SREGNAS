

from __future__ import annotations

from typing import Any


GIB = float(1024**3)
ARXIV_NUM_NODES = 169_343
ARXIV_NUM_EDGES = 2_315_598
ARXIV_INPUT_DIM = 128
ARXIV_NUM_CLASSES = 40
ARXIV_EFFECTIVE_CARD_BUDGET_GIB = 47.53
ARXIV_MEMORY_MODEL_CALIBRATION = (
    "d512_attention_boundary_sweep_160cell_right_censored_v1"
)
STATIC_GATE_PREDICATE = "arxiv_d512_attention_memory_boundary_v2"
ARXIV_V3_PREDICATE = "arxiv_calibrated_peak_memory_v3"
ARXIV_V3_NEAR_BUDGET_GUARD_FRACTION = 0.05
ARXIV_V3_NEAR_BUDGET_REJECT_THRESHOLD_GIB = (
    1.0 - ARXIV_V3_NEAR_BUDGET_GUARD_FRACTION
) * ARXIV_EFFECTIVE_CARD_BUDGET_GIB
ARXIV_V3_CAMPAIGN_STATUS = "BLOCKED_PENDING_RECALIBRATION_AND_EXPLICIT_APPROVAL"


MEASURED_CAMPAIGN_APPROVED_STATIC_GATE_PREDICATES = frozenset(
    {"arxiv_d512_attention_memory_boundary_v2"}
)





ARXIV_MEMORY_COEFFICIENTS = {
    "common_message_autograd": 3.982278082972,
    "gat_extra_l1": 1.032538544681,
    "gat_extra_l2": 1.000121415472,
    "attention_score_l1": 0.0,
    "attention_score_l2": 0.030287160372,
    "max_workspace_l1": 0.990695802426,
    "max_workspace_l2": 1.902696762612,
}


def _parameter_count(architecture: list[dict[str, Any]]) -> int:
    pass
    widths = [int(layer["hidden_dim"]) for layer in architecture]
    input_widths = [ARXIV_INPUT_DIM, widths[0]]
    count = 0
    for layer, in_width, out_width in zip(architecture, input_widths, widths):
        count += in_width * out_width + out_width
        if layer["encoding_op"] == "gat":
            count += 2 * out_width
        elif layer["encoding_op"] == "sym-gat":
            count += out_width
        if layer["aggregation_op"] == "mlp":
            count += 2 * (out_width * out_width + out_width)
    count += widths[-1] * ARXIV_NUM_CLASSES + ARXIV_NUM_CLASSES
    return count


def arxiv_peak_memory_prediction(
    architecture: list[dict[str, Any]],
) -> dict[str, Any]:
    pass
    in_domain = len(architecture) == 2 and all(
        layer.get("encoding_op") in {"gat", "sym-gat"} for layer in architecture
    )
    if not in_domain:
        return {
            "in_calibrated_domain": False,
            "predicted_peak_gib": None,
            "card_budget_gib": ARXIV_EFFECTIVE_CARD_BUDGET_GIB,
            "reject": False,
            "raw_model_reject": False,
            "near_budget_guard_reject": False,
            "guarded_reject": False,
            "v3_campaign_status": ARXIV_V3_CAMPAIGN_STATUS,
            "calibration": ARXIV_MEMORY_MODEL_CALIBRATION,
            "reason": "outside_two_attention_layer_calibration_domain",
        }

    full_edges = ARXIV_NUM_EDGES + ARXIV_NUM_NODES
    message_gib = [
        full_edges * int(layer["hidden_dim"]) * 4 / GIB for layer in architecture
    ]
    attention_score_gib = [
        full_edges * int(layer["num_heads"]) * 4 / GIB for layer in architecture
    ]
    graph_resident_gib = (
        ARXIV_NUM_NODES * ARXIV_INPUT_DIM * 4
        + 2 * ARXIV_NUM_EDGES * 8
        + ARXIV_NUM_NODES * 8
        + 3 * ARXIV_NUM_NODES
    ) / GIB
    activation_saved_gib = sum(
        ARXIV_NUM_NODES * int(layer["hidden_dim"]) * 4 / GIB
        for layer in architecture
        if layer.get("activation") != "none"
    )



    parameter_optimizer_gib = 16 * _parameter_count(architecture) / GIB

    coeff = ARXIV_MEMORY_COEFFICIENTS
    components = {
        "graph_resident_gib": graph_resident_gib,
        "activation_saved_gib": activation_saved_gib,
        "parameter_gradient_adam_gib": parameter_optimizer_gib,
        "common_message_autograd_gib": coeff["common_message_autograd"]
        * sum(message_gib),
        "gat_extra_l1_gib": coeff["gat_extra_l1"]
        * message_gib[0]
        * (architecture[0]["encoding_op"] == "gat"),
        "gat_extra_l2_gib": coeff["gat_extra_l2"]
        * message_gib[1]
        * (architecture[1]["encoding_op"] == "gat"),
        "attention_score_l1_gib": coeff["attention_score_l1"] * attention_score_gib[0],
        "attention_score_l2_gib": coeff["attention_score_l2"] * attention_score_gib[1],
        "max_workspace_l1_gib": coeff["max_workspace_l1"]
        * message_gib[0]
        * (architecture[0]["aggregation_op"] == "max"),
        "max_workspace_l2_gib": coeff["max_workspace_l2"]
        * message_gib[1]
        * (architecture[1]["aggregation_op"] == "max"),
    }
    predicted = float(sum(components.values()))
    raw_reject = predicted > ARXIV_EFFECTIVE_CARD_BUDGET_GIB
    guard_reject = predicted >= ARXIV_V3_NEAR_BUDGET_REJECT_THRESHOLD_GIB
    return {
        "in_calibrated_domain": True,
        "predicted_peak_gib": predicted,
        "card_budget_gib": ARXIV_EFFECTIVE_CARD_BUDGET_GIB,



        "reject": raw_reject,
        "raw_model_reject": raw_reject,
        "near_budget_guard_reject": guard_reject,
        "guarded_reject": raw_reject or guard_reject,
        "v3_campaign_status": ARXIV_V3_CAMPAIGN_STATUS,
        "calibration": ARXIV_MEMORY_MODEL_CALIBRATION,
        "components": components,
        "message_tensor_gib": message_gib,
        "attention_score_tensor_gib": attention_score_gib,
    }


def arxiv_v3_guarded_memory_gate_rejects(
    architecture: list[dict[str, Any]],
) -> bool:
    pass
    prediction = arxiv_peak_memory_prediction(architecture)
    return bool(prediction["guarded_reject"])


def assert_static_gate_predicate_approved(
    predicate: str = STATIC_GATE_PREDICATE,
) -> None:
    pass
    if predicate not in MEASURED_CAMPAIGN_APPROVED_STATIC_GATE_PREDICATES:
        raise RuntimeError(
            f"static gate predicate {predicate!r} is not explicitly approved "
            "for a measured campaign"
        )


def arxiv_static_memory_gate_rejects(
    architecture: list[dict[str, Any]],
) -> bool:
    pass
    return arxiv_d512_attention_memory_boundary_v2(architecture)


def arxiv_d512_attention_memory_boundary_v2(
    architecture: list[dict[str, Any]],
) -> bool:
    pass
    if len(architecture) != 2:
        return False
    if not all(
        int(layer.get("hidden_dim", 0)) == 512
        and layer.get("encoding_op") in {"gat", "sym-gat"}
        for layer in architecture
    ):
        return False
    first, second = architecture
    encodings = (first["encoding_op"], second["encoding_op"])
    if encodings == ("gat", "gat"):
        return True
    if not all(int(layer["num_heads"]) >= 16 for layer in architecture):
        return False
    if second["aggregation_op"] == "max":
        return True
    return first["aggregation_op"] == "max" and "gat" in encodings



arxiv_categorical_memory_gate_rejects_v2 = arxiv_d512_attention_memory_boundary_v2
