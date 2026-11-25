# WeatherGenerator latent subsetter

Export raw WeatherGenerator hindcast Zarr ZIPs to a regional, xarray-compatible
Zarr v3 dataset. This package is independent of the latent decoder and requires
Python 3.12.

```bash
uv sync --project packages/latent_subset
uv run --project packages/latent_subset weathergen-subset \
    /path/to/raw-hindcast /path/to/new-europe.zarr \
    --south 34 --north 72 --west -12 --east 60
```

The source can be an archive, year or single run directory containing raw
hindcast manifests. The output must be a new directory separate from the source.
The example selects a broad Europe bounding box, including areas outside Europe.
Bounds include token centres on their edges; they do not select every cell
intersecting the box. Longitude ranges can cross the dateline.

## Output

The `tokens` variable has dimensions `(sample, lead, cell, channel)` and keeps
the source dtype and values. Every initialization in the manifests is retained,
including duplicates: `sample` is a unique integer index, and
`initialization_time`, `source_run`, `source_zip`, and `source_position` identify
each source. `lead` contains the numeric source step IDs. The `cell` coordinate
contains global nested HEALPix IDs in their original order; `latitude` and
`longitude` are the tokenizer's geographic cell centres.

The exporter writes one token shard per sample, containing all leads and cells.
Logical token chunks have shape `(1, 1, 64, channels)`, with all channels in
each chunk. The cell extent of a shard is rounded up to a multiple of 64, but
the array's cell dimension contains only selected cells. For the example box,
488 cells are exposed and the shard extent is 512. Data are compressed with
Zstandard, and raw token values are not normalized, averaged or cast.

The dataset also retains `register_tokens`, `valid_interval`,
`prediction_times`, `prediction_coords`, `prediction_geoinfo`, and
`class_token`. `class_token_present` distinguishes a missing source array from
an empty or present one; values for absent class tokens are zero-filled and
must not be treated as data. `prediction_coords` reproduces the stored values,
which can be mislabeled: use the `latitude` and `longitude` coordinates for
geographic locations. Time arrays are encoded as CF-compatible nanosecond
datetimes; values that cannot be represented without loss are rejected.

The `provenance/` group contains copies of the original manifests and non-ZIP
run files, plus a `.metadata.zip` for each source ZIP. These metadata archives
preserve the original Zarr metadata and other non-chunk ZIP entries, including
source group and array attributes. The manifest checksums refer to **source**
files, not the new dataset. Unknown source arrays or inconsistent shapes, dtypes,
or lead steps fail rather than being silently discarded.

Open only successfully completed outputs:

```python
import xarray as xr

ds = xr.open_zarr("/path/to/new-europe.zarr", consolidated=False)
if ds.attrs["export_status"] != "complete":
    raise RuntimeError("The export is incomplete")

local_tokens = ds.tokens.isel(sample=0, lead=0, cell=slice(0, 64)).values
```

The output is created before data writing starts and has `export_status` set to
`in_progress` until the export succeeds. For the serial command, use a fresh
destination after an interruption or failure. The source is not modified.
One sample is buffered while writing, so memory usage grows with the selected
region; a full-global export needs substantially more memory than Europe.

## Slurm: one dataset across years

Install the package first as above. Run the wrapper with `bash` (it submits
the jobs itself). This example exports the 45 Norway cells whose centres fall
inside the box:

```bash
bash packages/latent_subset/submit_subset.sh \
    /ec/weathergen/latents/sdnbfpve-v1-raw "$SCRATCH/hindcast-norway-single.zarr" \
    2016 2024 --south 58 --north 72 --west 4 --east 32
```

For a broad Europe subset in one store, use a fresh output path and pass
`--south 34 --north 72 --west -12 --east 60` instead.

The wrapper initializes one store, submits one array task per year (two at a
time), and finalizes only after the array succeeds. Array tasks derive their
year from `SLURM_ARRAY_TASK_ID` and write only the sample range assigned to
that year. Open the output path itself once its `export_status` is `complete`.
Job IDs and the log directory are printed on submission. The destination must
not exist. Adjust the script's Slurm account, partition, QOS and 16-hour worker
limit for your site. Scratch is temporary; use persistent storage for retention.

Unlike the serial export, the Slurm workflow can retry an unfinished year
without rewriting successful years. After the failed task has stopped, run
`weathergen-subset write-year OUTPUT YEAR` in a compute allocation (or submit
it as a batch job). Then run `weathergen-subset finalize OUTPUT` once all years
have succeeded. If initialization failed, start again with a fresh destination.
