"""Module for helper utils which don't belong to any other modules."""

from __future__ import annotations

import random
import string
from typing import Optional, Union


def get_rnd_str(k: int, seed: Optional[Union[str, int, float, bytes]] = None) -> str:
    """Creates a random ascii+digits string with length=`k`.

    Most tmp-file tools are using a string-length of 10.

    Args:
        k: Length of random string.
        seed: Seed.

    Returns:
        Random generated string.
    """
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))
