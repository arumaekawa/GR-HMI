from .pq_modules import (
    HippocampusPQLossDifference,
    HippocampusPQLowPerplexity,
    HippocampusPQNearestKmeans,
    HippocampusPQRandom,
    HippocampusPQRandomReal,
)
from .selective_modules import (
    HippocampusLossDifference,
    HippocampusLowPerplexity,
    HippocampusNearestKmeans,
    HippocampusRandom,
)

HIPPOCAMPUS_MODULE_CLASSES = {
    # without PQ
    "random": HippocampusRandom,
    "nearest_kmeans": HippocampusNearestKmeans,
    "loss_diff": HippocampusLossDifference,
    "low_ppl": HippocampusLowPerplexity,
    # with PQ
    "random_pq": HippocampusPQRandom,
    "nearest_kmeans_pq": HippocampusPQNearestKmeans,
    "loss_diff_pq": HippocampusPQLossDifference,
    "low_ppl_pq": HippocampusPQLowPerplexity,
    # use real samples
    "random_pq_real": HippocampusPQRandomReal,
}

__all__ = [
    "HIPPOCAMPUS_MODULE_CLASSES",
    "HippocampusRandom",
    "HippocampusNearestKmeans",
    "HippocampusLossDifference",
    "HippocampusLowPerplexity",
    "HippocampusPQRandom",
    "HippocampusPQNearestKmeans",
    "HippocampusPQLossDifference",
    "HippocampusPQLowPerplexity",
    "HippocampusPQRandomReal",
]
