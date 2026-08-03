# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Per-stream latent-level assignment (Step 3, sub-step 1).

This module owns the ONE policy decision that multi-resolution encoding rests on:
which latent level each stream is encoded at, and which level(s) each stream
reads from at decode. It is deliberately the only place where a stream name meets
a healpix level, so that "add another stream" stays pure configuration and the
pyramid / cascade code never mentions streams (the extensibility rule).

Two mappings are produced, both keyed by stream name:

  encode_level[stream]  -> the single latent level this stream writes into.
                           A coarse stream (e.g. ERA5) writes the coarse level;
                           a fine stream (e.g. MEPS) writes the fine level.

  decode_levels[stream] -> the ordered list of latent levels this stream reads at
                           the decode/readout stage. A coarse stream reads only its
                           own level; a fine stream reads coarse+fine so its
                           prediction is conditioned on the large-scale context as
                           well as local detail.

Resolution policy
-----------------
Config may specify the level per stream explicitly, OR give each stream a nominal
resolution (in km or degrees) and let the SNAPPING policy quantise it onto the
fixed level ladder of the pyramid. Snapping keeps the number of levels small and
fixed no matter how many streams are added: each stream is attached to the finest
ladder level whose cell size is still >= the stream's resolution (i.e. the ladder
level that does not over-resolve the stream). This guarantees >= ~1 obs per cell
for that stream, which is exactly what avoids the sparse-write latent artefacts.

Backward compatibility
----------------------
With a single-level pyramid, every stream is assigned that one level for both
encode and decode -- identical to the current single-latent behaviour.
"""

import logging

_logger = logging.getLogger(__name__)

# Approximate HEALPix cell edge length in km at the equator, per level.
# cell_km(level) ~ 58.6 deg / 2**level * 111 km/deg  (58.6 is the codebase's own
# cell-diagonal-in-degrees constant used elsewhere, e.g. Domain.bbox_point_mask).
_DEG_PER_CELL_CONST = 58.6
_KM_PER_DEG = 111.195


def cell_km(level: int) -> float:
    """Approximate healpix cell size in km at `level`."""
    return (_DEG_PER_CELL_CONST / (2 ** level)) * _KM_PER_DEG


def _resolution_km(stream_cfg: dict) -> float | None:
    """
    Extract a nominal resolution in km from a stream config, if present.

    Accepts either `resolution_km` directly, or `resolution_deg` (converted).
    Returns None if the stream does not declare a resolution (then an explicit
    `latent_level` is required instead).
    """
    if "resolution_km" in stream_cfg:
        return float(stream_cfg["resolution_km"])
    if "resolution_deg" in stream_cfg:
        return float(stream_cfg["resolution_deg"]) * _KM_PER_DEG
    return None


def snap_to_ladder(resolution_km: float, ladder: list[int]) -> int:
    """
    Snap a stream resolution onto the fixed level ladder.

    Policy: choose the FINEST ladder level whose cell size is still >= the stream
    resolution -- i.e. the ladder level that does not over-resolve the stream. If
    the stream is coarser than every ladder level (its resolution exceeds even the
    coarsest cell), it snaps to the coarsest level. If it is finer than every
    ladder level, it snaps to the finest.

    `ladder` is the list of configured levels, ascending (coarse -> fine).
    """
    assert len(ladder) >= 1
    # cell sizes descend as level ascends
    chosen = ladder[0]  # coarsest as default (stream coarser than everything)
    for lvl in ladder:
        if cell_km(lvl) >= resolution_km:
            chosen = lvl
        else:
            break
    return chosen


def assign_stream_levels(
    cf, ladder: list[int]
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """
    Build encode_level and decode_levels for every stream.

    Resolution order for a stream's ENCODE level:
      1. explicit `latent_level` in the stream config wins;
      2. else snap the stream's declared resolution onto the ladder;
      3. else (single-level ladder) use that one level;
      4. else error -- ambiguous, ask the user to declare one.

    DECODE levels for a stream:
      - explicit `decode_levels` list in the stream config wins;
      - else a coarse stream (encode == coarsest) reads only its own level;
      - else a fine stream reads [coarsest .. its encode level] so it is
        conditioned on all coarser context plus its own detail.

    Returns (encode_level, decode_levels), both keyed by stream name.
    """
    streams = cf.streams
    ladder = sorted(ladder)
    single = len(ladder) == 1

    encode_level: dict[str, int] = {}
    decode_levels: dict[str, list[int]] = {}

    for name, scfg in streams.items():
        scfg = scfg if isinstance(scfg, dict) else dict(scfg)

        # --- encode level ---
        if "latent_level" in scfg:
            lvl = int(scfg["latent_level"])
            assert lvl in ladder, (
                f"stream {name!r}: latent_level={lvl} is not in the configured "
                f"level ladder {ladder}."
            )
        elif single:
            lvl = ladder[0]
        else:
            res = _resolution_km(scfg)
            if res is None:
                raise ValueError(
                    f"stream {name!r}: with a multi-level ladder you must give either "
                    f"`latent_level` or a resolution (`resolution_km`/`resolution_deg`) "
                    f"so it can be snapped onto {ladder}."
                )
            lvl = snap_to_ladder(res, ladder)
        encode_level[name] = lvl

        # --- decode levels ---
        if "decode_levels" in scfg:
            dl = [int(x) for x in scfg["decode_levels"]]
            for x in dl:
                assert x in ladder, (
                    f"stream {name!r}: decode_levels contains {x}, not in ladder {ladder}."
                )
            decode_levels[name] = sorted(dl)
        elif single:
            decode_levels[name] = [ladder[0]]
        elif lvl == ladder[0]:
            # coarse stream: reads only its own (coarsest) level
            decode_levels[name] = [lvl]
        else:
            # fine stream: reads coarse context up to and including its own level
            decode_levels[name] = [x for x in ladder if x <= lvl]

    for name in streams:
        _logger.info(
            "stream %-12s encode@hl%d  decode@%s",
            name, encode_level[name],
            ",".join(f"hl{x}" for x in decode_levels[name]),
        )

    return encode_level, decode_levels
