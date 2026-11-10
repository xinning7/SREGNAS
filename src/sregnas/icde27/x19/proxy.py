
import torch

ENV = 'SREGNAS_X19_PREBUILD_BATCHES'


def prebuild(index, train_nodes, epochs, batch_targets, seed):
    from sregnas.training.fixed_proxy import APTargetDraws



    cpu_nodes = getattr(index, '_x19_cpu_train_nodes', None)
    if cpu_nodes is None:
        cpu_nodes = train_nodes.cpu()
        index._x19_cpu_train_nodes = cpu_nodes
    draws = APTargetDraws(cpu_nodes, batch_targets, seed)
    targets = torch.stack([draws.draw() for _ in range(epochs)]).to(train_nodes.device)
    targets = targets.sort(dim=1).values
    return [(row, index.slice(row)) for row in targets]


def prepared_step(plan, step):
    pass
    return plan[step]
