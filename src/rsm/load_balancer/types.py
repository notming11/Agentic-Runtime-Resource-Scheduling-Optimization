from enum import Enum

class RequestSize(Enum):
    """Request size classification based on prefill tokens."""
    SMALL = "small"  # <= 2048 tokens - load balance
    LARGE = "large"  # > 2048 tokens - use locality