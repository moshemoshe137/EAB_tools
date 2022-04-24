"""Default values used throughout the package"""

from collections import defaultdict

eab_rc = defaultdict(
    default_factory=lambda: None,  # By default, return None
    hash_len=7,  # int | None
)
