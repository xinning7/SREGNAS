
import hashlib


def native_rng_assertions(segments, events):
    import torch
    from sregnas.training.minibatch import _hash_seed
    starts=sorted([s for s in segments if s["event"]=="segment_start"],key=lambda s:s["perf_s"])
    resets=[e for e in events if e["event"]=="native_rng_reset"]
    training=[e for e in events if e["event"]=="native_training_start"]
    previous={}; checked=[]
    for segment in starts:
        sid=segment["segment_id"]
        init=[r for r in resets if r["segment_id"]==sid and r["scope"]=="initialization"]
        part=[r for r in resets if r["segment_id"]==sid and r["scope"]=="partition_training"]
        train=[r for r in training if r["segment_id"]==sid]
        rebuild=previous.get(segment["worker"])!=segment["fingerprint"]
        previous[segment["worker"]]=segment["fingerprint"]
        init_ok=(len(init)==1 if rebuild else len(init) in (0,1)) and len(train)==1
        init_ok=init_ok and all(segment["perf_s"]<=r["perf_s"]<train[0]["perf_s"] for r in init)
        part_ok=len(part)==len(train)==1 and part[0]["perf_s"]<train[0]["perf_s"]
        for reset in init+part:
            inputs=[segment["fingerprint"],"search",0]
            if reset["scope"]=="partition_training": inputs.append(segment["partition_id"])
            expected=_hash_seed(*inputs)
            cpu_hash=hashlib.sha256(torch.Generator(device="cpu").manual_seed(expected).get_state().numpy().tobytes()).hexdigest()
            part_ok=part_ok and reset["seed_inputs"]==inputs and reset["seed"]==expected and reset["torch_cpu_state_sha256"]==cpu_hash and bool(reset["torch_cuda_state_sha256"])
        checked.append(dict(segment_id=sid, initialization=bool(init_ok), partition_rng=bool(part_ok)))
    return dict(native_initialization_rng=bool(checked) and all(r["initialization"] for r in checked),
                native_partition_rng=bool(checked) and all(r["partition_rng"] for r in checked)), checked
