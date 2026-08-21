"""Device selection.

Written on a laptop without a usable GPU, trained on a workstation with one, so
nothing hard-codes a device. Modules take `device` and default to this.
"""

import torch


def default_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
