

from sregnas.baselines.graphnap.common import ParallelCandidateWorker
from .reference import ArchParallelWorker


class Worker(ParallelCandidateWorker, ArchParallelWorker):
    pass
