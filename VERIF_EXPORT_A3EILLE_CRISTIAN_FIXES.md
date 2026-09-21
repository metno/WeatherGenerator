# Verif export changes: combining `a3eille` and Cristian behavior

This note documents the changes made on top of `a3eille/verif_step0_20260917` after applying the stash, and explains why the behavior differed from `cristianl/metno-develop-20260807`.

The goal was to keep the useful stashed `a3eille` behavior, while taking the better parts of Cristian's branch for:

1. forecast reference time handling;
2. reshaping flattened zarr data for Verif export.

No commit was made.

## Output directories discussed

The comparison was between:

- `./y98t6d5uMET_Norway_aws`
  - produced from `a3eille/verif_step0_20260917` plus the stash;
- `./y98t6d5uMET_Norway_aws_chr`
  - produced from `cristianl/metno-develop-20260807`.

The relevant output command was:

```bash
id=y98t6d5u
s=MET_Norway_aws
uv run --offline export \
  --run-id $id \
  --stream $s \
  --output-dir ./${id}${s} \
  --format verif \
  --channels 10si 2t 2r sp tp \
  --method nearest \
  --verif-template ./verif/%S/%V/%R_%S_%V_%M_%D.nc \
  --obs /home/amelien/data/wg/metno_observations_v5.nc
```

## Important correction about the branches

The clean `a3eille/verif_step0_20260917` branch did **not** match the output files in `./y98t6d5uMET_Norway_aws`.

Those files were produced using `a3eille/verif_step0_20260917` **plus a stash**:

```text
stash@{0}: WIP on verif_step0_20260917: 9379ac15 fixed for experiment with missing step 0
```

That stash changed:

```text
config/evaluate/config_zarr2verif.yaml
packages/evaluate/src/weathergen/evaluate/export/parsers/verif_parser.py
```

This matters because the stash added the v5-style MET Norway variable names and `2r` support.

## What was already good in the stashed `a3eille` code

The stashed `a3eille` code already had important fixes that should be kept:

- support for v5-style observation names:

  ```yaml
  2t   -> obs_name: 2t
  sp   -> obs_name: sp
  tp   -> obs_name: tp
  10si -> obs_name: 10si
  2r   -> obs_name: 2r
  ```

- support for `2r`;
- support for `10si` directly;
- configurable precipitation handling using `obs_name`;
- rank-aware export logic.

In particular, the stashed precipitation helper was better adapted to `metno_observations_v5.nc` than Cristian's branch, because it passed the configured `obs_name` into `compute_precip`.

## Point 1: forecast reference time

### What differed

The stashed `a3eille` code allowed:

```text
--init-time-reference source_start
--init-time-reference source_end
```

but its default was:

```text
source_start
```

So if the source/conditioning window was:

```text
2023-06-01 06:00 -> 2023-06-01 12:00
```

then the default forecast reference time was:

```text
2023-06-01 06:00
```

Cristian's branch instead used:

```text
forecast_reference_time = source_end
```

so:

```text
forecast_reference_time = 2023-06-01 12:00
```

### Why it matters

The Verif files store:

```text
time = forecast_reference_time
leadtime = valid_time - forecast_reference_time
```

So changing the reference time changes every leadtime.

Example:

```text
valid_time = 2023-06-01 18:00
```

If reference time is `source_start = 06:00`:

```text
leadtime = 12h
```

If reference time is `source_end = 12:00`:

```text
leadtime = 6h
```

The same physical valid time gets a different leadtime coordinate.

### Why `source_end` is preferred

For a model that consumes a source/conditioning window and then predicts after it, the forecast normally starts at the **end** of the source window.

Example:

```text
source window: 06:00, 07:00, ..., 12:00
forecast starts: 12:00
leadtime +6h: 18:00
```

So `source_end` is the more natural and usually more correct forecast reference time.

### What was changed

In `packages/evaluate/src/weathergen/evaluate/export/export_core.py`, the default changed from:

```python
init_time_reference = kwargs.get("init_time_reference", "source_start")
```

to:

```python
init_time_reference = kwargs.get("init_time_reference", "source_end")
```

In `packages/evaluate/src/weathergen/evaluate/export/export_inference.py`, the CLI default changed from:

```python
default="source_start"
```

to:

```python
default="source_end"
```

The help text was also updated.

### Additional safety fix

The code previously passed:

```python
source_interval_end=init_time
```

This was changed to:

```python
source_interval_end=source_end
```

This is important because `source_interval_end` is used to compute:

```python
zarr_dt = source_interval_end - source_interval_start
```

That interval is used for precipitation accumulation.

If someone explicitly used:

```bash
--init-time-reference source_start
```

then `init_time == source_start`. If `source_interval_end=init_time`, the interval would become:

```text
source_start - source_start = 0
```

which would break precipitation accumulation.

After the change:

- `ref_time` controls the Verif forecast reference time;
- `source_interval_start/source_interval_end` always describe the real source window.

## Point 2: reshaping flattened zarr data

### What differed

The zarr data arrives with a flat `ipoint` dimension.

Conceptually, `ipoint` may contain:

```text
valid_time × spatial_point
```

For example, with 2 valid times and 3 spatial points:

```text
ipoint 0: time0, point0
ipoint 1: time0, point1
ipoint 2: time0, point2
ipoint 3: time1, point0
ipoint 4: time1, point1
ipoint 5: time1, point2
```

This is time-major order:

```text
t0 all points, then t1 all points, then t2 all points, ...
```

### What stashed `a3eille` did

The stashed `a3eille` code split each result by valid time:

```python
unique_times = np.sort(np.unique(result.valid_time.values))

for vt in unique_times:
    sub = result.sel(channel=self.channels, valid_time=vt)
    sub = sub.assign_coords(ipoint=np.arange(sub.sizes["ipoint"]))
    sub = self.reshape(sub)
    da_fs.append(sub)
```

This is a workaround. It avoids directly handling a flattened `(valid_time × point)` axis.

### What Cristian's branch did

Cristian's branch kept the full result together and explicitly reconstructed the intended shape:

```python
n_vt = number of valid times
n_points = n_ipoint // n_vt
```

Then it reshaped:

```text
ipoint -> (valid_time, point) -> (point, valid_time)
```

So the output becomes:

```text
ncells × valid_time
```

### Why it matters

The Verif export interpolates from model grid cells to observation stations.

That means every value must stay attached to the correct latitude/longitude.

If the flattened `ipoint` axis is interpreted incorrectly, then this can happen:

```text
station A gets nearest grid point X,
but the value attached to grid point X came from another time or another point.
```

That can produce very different plotted values even when using the same nearest-neighbour method.

### What was changed

In `packages/evaluate/src/weathergen/evaluate/export/parsers/verif_parser.py`, the split-by-valid-time loop was removed.

Before:

```python
for vt in unique_times:
    sub = result.sel(channel=self.channels, valid_time=vt)
    ...
    da_fs.append(sub)
```

Now:

```python
result = result.sel(channel=self.channels)
result = self.preprocess(result)
result = self.reshape(result)
da_fs.append(result)
```

Then `reshape` now explicitly unflattens:

```python
n_vt = np.unique(data["valid_time"].values).size
n_ipoint = data.sizes["ipoint"]
n_points = n_ipoint // n_vt

reshaped = arr.reshape((n_vt, n_points) + arr.shape[1:])
return reshaped.transpose(1, 0, ...)
```

So values become:

```text
ncells × valid_time
```

### Safety checks

The new reshape checks:

```python
lat does not change across valid_time for the same point
lon does not change across valid_time for the same point
valid_time is consistent across points
```

These checks are useful because if the assumed flattened ordering is wrong, the code should fail loudly instead of silently writing wrong Verif files.

## Added debug prints from Cristian's branch

Cristian's grid diagnostics were added in `verif_parser.py`, just before the grid consistency assertion between forecast steps.

They print:

```text
lat constant across valid_time now: ...
grid shapes: ...
lat dims: ...
grid equal: ...
n differing rows: ...
same set after re-sort: ...
```

### How to interpret them

If you see:

```text
grid equal: True
```

then forecast steps use the same grid in the same order.

If you see:

```text
grid equal: False
same set after re-sort: True
```

then the forecast steps have the same grid points, but in a different order. That is dangerous because values may be matched to the wrong coordinates unless reordered.

If you see:

```text
grid equal: False
same set after re-sort: False
```

then the forecast steps are on genuinely different point sets. That probably indicates masking, cropping, or inconsistent output.

## Final combined behavior

The current branch now keeps useful pieces from both implementations.

From Cristian's branch:

- forecast reference time defaults to `source_end`;
- flattened `(valid_time × point)` data is reconstructed explicitly;
- grid diagnostics are available.

From stashed `a3eille`:

- v5 observation variable mapping is kept;
- `2r` support is kept;
- `10si` support is kept;
- configurable `tp` precipitation handling via `obs_name` is kept;
- rank-aware export logic is kept.

This should make the current branch a better candidate than either original branch alone for:

```text
MET_Norway_aws + metno_observations_v5.nc + Verif export
```

## Validation performed

The following checks passed:

```bash
python -m py_compile \
  packages/evaluate/src/weathergen/evaluate/export/parsers/verif_parser.py \
  packages/evaluate/src/weathergen/evaluate/export/export_core.py \
  packages/evaluate/src/weathergen/evaluate/export/export_inference.py
```

```bash
git diff --check
```

A dependency-free sanity check of the unflattening logic also passed.

