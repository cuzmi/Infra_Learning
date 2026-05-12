"""
Compatibility exports for the main MoE building blocks.

The implementation is split by responsibility:
    outputs.py: dataclasses returned by the modules
    router.py: top-k routing decisions
    experts.py: feed-forward expert network
    layer.py: dispatch tokens to experts and combine outputs
"""

from .experts import FeedForwardExpert
from .layer import MoELayer
from .outputs import MoEOutput, RouterOutput
from .router import TopKRouter

__all__ = [
    "FeedForwardExpert",
    "MoELayer",
    "MoEOutput",
    "RouterOutput",
    "TopKRouter",
]

