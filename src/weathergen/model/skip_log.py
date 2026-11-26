# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Log the places where the model SKIPS a stream's module on this rank (no data).

With FSDP every rank must run the same modules: a module skipped on one rank only puts
the collectives out of step and training hangs (NCCL watchdog timeout after 10 min).
These lines show which rank skipped which stream, and why, just before such a hang.

Each (stream, reason) is logged for its first `first` occurrences in a process, then
every `every`-th one (a stream that is skipped on purpose every step, e.g. a target-only
stream without source, does not flood the log).
"""

import logging

import torch

_logger = logging.getLogger(__name__)
_counts: dict[tuple[str, str], int] = {}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def log_skip(where: str, stream: str, reason: str, first: int = 20, every: int = 1000) -> None:
    key = (where, stream, reason)
    n = _counts.get(key, 0) + 1
    _counts[key] = n
    if n <= first or n % every == 0:
        _logger.warning(
            "[skip] rank %d: %s %s skipped (%s), %d times so far", _rank(), where, stream,
            reason, n,
        )
