"""Export raw hindcast ZIPs to a regional, xarray-compatible Zarr dataset."""

import argparse
import fcntl
import hashlib
import json
import logging
import pathlib
import re
import shutil
import sys
import zipfile

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import ZstdCodec
from zarr.storage import ZipStore

from weathergen_subset.geometry import N_CELLS, NSIDE, cell_centres

LOG = logging.getLogger(__name__)
CELL_CHUNK = 64
ARRAY_PATH = re.compile(
    r"\d+/latent/\d+/(?:prediction/(?:data|coords|times|geoinfo)"
    r"|register_tokens|class_token|valid_interval)/zarr\.json"
)


def select_cells(south, north, west, east):
    """Select centres within inclusive bounds, allowing dateline crossings."""
    bounds = np.asarray([south, north, west, east], dtype=float)
    if not np.isfinite(bounds).all():
        raise ValueError("box bounds must be finite")
    if not -90 <= south <= north <= 90:
        raise ValueError("require -90 <= south <= north <= 90")
    if not (-180 <= west <= 360 and -180 <= east <= 360):
        raise ValueError("longitudes must be in [-180, 360]")
    if abs(east - west) > 360:
        raise ValueError("longitude span cannot exceed 360 degrees")
    lat, lon = cell_centres()
    full_longitude = abs(east - west) == 360
    west, east = (west + 180) % 360 - 180, (east + 180) % 360 - 180
    longitude = (lon >= west) & (lon <= east)
    if west > east:
        longitude = (lon >= west) | (lon <= east)
    cells = np.flatnonzero((lat >= south) & (lat <= north) & (longitude | full_longitude))
    if not len(cells):
        raise ValueError("box contains no token centres; enlarge the box")
    return cells


def _file_name(name, manifest):
    if (
        not isinstance(name, str)
        or pathlib.Path(name).name != name
        or name in (".", "..", "MANIFEST.json")
    ):
        raise ValueError(f"{manifest}: invalid file name {name!r}")


def _discover(source, archive_root=None):
    manifests = sorted(
        path for depth in range(3) for path in source.glob("*/" * depth + "MANIFEST.json")
    )
    if not manifests:
        raise ValueError(f"{source}: no raw hindcast MANIFEST.json files found")
    runs, samples = [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        files, inits = manifest["files"], manifest["inits"]
        if not isinstance(files, dict) or not isinstance(inits, dict):
            raise ValueError(f"{path}: files and inits must be mappings")
        for name in files:
            _file_name(name, path)
            if not (path.parent / name).is_file():
                raise FileNotFoundError(f"{path}: missing file {name}")
        if not set(inits).issubset(files):
            raise ValueError(f"{path}: initialization ZIP missing from files")
        relative = path.parent.relative_to(archive_root or source)
        runs.append((path, relative, manifest))
        for name, bases in inits.items():
            if not isinstance(bases, list):
                raise ValueError(f"{path}: initializations for {name} must be a list")
            for position, base in enumerate(bases):
                try:
                    original = np.datetime64(base)
                    time = original.astype("datetime64[ns]")
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"{path}: invalid initialization time {base!r}") from error
                if np.isnat(time) or time.astype(original.dtype) != original:
                    raise ValueError(f"{path}: invalid initialization time {base!r}")
                samples.append((path.parent / name, position, time, str(relative), name))
    if not samples:
        raise ValueError(f"{source}: no initializations in raw hindcast manifests")
    return runs, samples


def _sample_groups(root, bases, path):
    try:
        groups = sorted(root.group_keys(), key=int)
    except ValueError as error:
        raise ValueError(f"{path}: sample group names must be integers") from error
    if len(groups) != len(bases) or list(root.array_keys()):
        raise ValueError(f"{path}: sample groups do not match manifest initializations")
    return groups


def _lead_steps(group, path):
    if set(group.group_keys()) != {"latent"} or list(group.array_keys()):
        raise ValueError(f"{path}: expected a latent group and no other sample arrays")
    latent = group["latent"]
    try:
        names = sorted(latent.group_keys(), key=int)
        steps = [int(name) for name in names]
    except ValueError as error:
        raise ValueError(f"{path}: lead group names must be integers") from error
    if not steps or len(set(steps)) != len(steps) or list(latent.array_keys()):
        raise ValueError(f"{path}: invalid or empty lead groups")
    return names, steps


def _state_arrays(state, path):
    if set(state.group_keys()) != {"prediction"}:
        raise ValueError(f"{path}: expected one prediction group")
    arrays = set(state.array_keys())
    if not {"register_tokens", "valid_interval"}.issubset(arrays) or arrays - {
        "register_tokens",
        "class_token",
        "valid_interval",
    }:
        raise ValueError(f"{path}: unexpected or missing state arrays: {sorted(arrays)}")
    prediction = state["prediction"]
    expected = {"data", "coords", "times", "geoinfo"}
    if set(prediction.array_keys()) != expected or list(prediction.group_keys()):
        raise ValueError(f"{path}: unexpected or missing prediction arrays")
    return {
        "tokens": prediction["data"],
        "prediction_coords": prediction["coords"],
        "prediction_times": prediction["times"],
        "prediction_geoinfo": prediction["geoinfo"],
        "register_tokens": state["register_tokens"],
        "valid_interval": state["valid_interval"],
        "class_token": state["class_token"] if "class_token" in arrays else None,
    }


def _schema(state, path):
    arrays = _state_arrays(state, path)
    tokens = arrays["tokens"]
    if len(tokens.shape) != 3 or tokens.shape[0] != N_CELLS or tokens.shape[2] != 1:
        raise ValueError(f"{path}: unexpected token shape {tokens.shape}")
    channels = tokens.shape[1]
    if not channels:
        raise ValueError(f"{path}: token channel dimension is empty")
    expected_shapes = {
        "prediction_coords": (N_CELLS, 2),
        "prediction_times": (N_CELLS,),
        "valid_interval": (2,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path}: unexpected {name} shape {arrays[name].shape}")
    geo = arrays["prediction_geoinfo"]
    registers = arrays["register_tokens"]
    class_token = arrays["class_token"]
    if len(geo.shape) != 2 or geo.shape[0] != N_CELLS:
        raise ValueError(f"{path}: unexpected prediction_geoinfo shape {geo.shape}")
    if len(registers.shape) != 2 or registers.shape[1] != channels:
        raise ValueError(f"{path}: unexpected register_tokens shape {registers.shape}")
    if class_token is not None and (
        len(class_token.shape) != 2 or class_token.shape[1] != channels
    ):
        raise ValueError(f"{path}: unexpected class_token shape {class_token.shape}")
    for name in ("prediction_times", "valid_interval"):
        if arrays[name].dtype.kind != "M":
            raise ValueError(f"{path}: {name} must have datetime64 dtype")
    return {
        name: (array.shape, np.dtype(array.dtype))
        if array is not None
        else ((0, channels), np.dtype(tokens.dtype))
        for name, array in arrays.items()
    }


def _definitions(schema, n_cells):
    channels = schema["tokens"][0][1]
    n_registers = schema["register_tokens"][0][0]
    n_class = schema["class_token"][0][0]
    n_geoinfo = schema["prediction_geoinfo"][0][1]
    return {
        "tokens": (("cell", "channel"), (n_cells, channels)),
        "prediction_coords": (("cell", "coordinate"), (n_cells, 2)),
        "prediction_times": (("cell",), (n_cells,)),
        "prediction_geoinfo": (("cell", "geoinfo"), (n_cells, n_geoinfo)),
        "register_tokens": (("register", "channel"), (n_registers, channels)),
        "class_token": (("class_token_index", "channel"), (n_class, channels)),
        "valid_interval": (("interval_bound",), (2,)),
        "class_token_present": ((), ()),
    }


def _dataset(samples, steps, cells, schema, selection, source):
    n_samples, n_leads, n_cells = len(samples), len(steps), len(cells)
    channels = schema["tokens"][0][1]
    definitions = _definitions(schema, n_cells)
    padded_cells = ((n_cells + CELL_CHUNK - 1) // CELL_CHUNK) * CELL_CHUNK
    variables, encoding = {}, {}
    for name, (dims, rest) in definitions.items():
        shape = (n_samples, n_leads, *rest)
        chunks = (
            1,
            1,
            *(
                CELL_CHUNK if dim == "cell" else max(1, size)
                for dim, size in zip(dims, rest, strict=True)
            ),
        )
        shards = (
            1,
            n_leads,
            *(
                padded_cells if dim == "cell" else max(1, size)
                for dim, size in zip(dims, rest, strict=True)
            ),
        )
        dtype = np.dtype(bool) if name == "class_token_present" else schema[name][1]
        if dtype.kind == "M":
            dtype = np.dtype("datetime64[ns]")
        variables[name] = (("sample", "lead", *dims), da.empty(shape, chunks=shards, dtype=dtype))
        encoding[name] = {
            "chunks": chunks,
            "shards": shards,
            "compressors": [ZstdCodec(level=0, checksum=False)],
            "_FillValue": None,
        }
        if dtype.kind == "M":
            encoding[name].update(
                dtype="int64",
                units="nanoseconds since 1970-01-01",
                calendar="proleptic_gregorian",
            )
    lat, lon = cell_centres()
    ds = xr.Dataset(
        variables,
        coords={
            "sample": np.arange(n_samples),
            "initialization_time": ("sample", np.asarray([s[2] for s in samples])),
            "source_run": ("sample", np.asarray([s[3] for s in samples], dtype=object)),
            "source_zip": ("sample", np.asarray([s[4] for s in samples], dtype=object)),
            "source_position": ("sample", np.asarray([s[1] for s in samples])),
            "lead": np.asarray(steps),
            "cell": cells,
            "latitude": ("cell", lat[cells]),
            "longitude": ("cell", lon[cells]),
            "channel": np.arange(channels),
            "coordinate": np.asarray(["latitude", "longitude"], dtype=object),
            "interval_bound": np.asarray(["start", "end"], dtype=object),
        },
        attrs={
            "export_status": "in_progress",
            "source_directory": str(source),
            "spatial_subset": {
                "bounds": selection,
                "selection": "inclusive geographic token centres",
                "nside": NSIDE,
                "order": "nested",
                "tokenizer_longitude_shift_degrees": 180,
            },
        },
    )
    ds["prediction_coords"].attrs["description"] = (
        "Original stored coordinates; these may be geographically mislabeled. "
        "Use latitude and longitude coordinates for geographic token centres."
    )
    ds["class_token_present"].attrs["description"] = (
        "Whether the source state contained a class_token array."
    )
    return ds, encoding


def _save_zip_metadata(source, destination):
    with (
        zipfile.ZipFile(source) as src,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            name = info.filename
            if name.endswith("zarr.json"):
                data = src.read(info)
                metadata = json.loads(data)
                if metadata.get("zarr_format") != 3:
                    raise ValueError(f"{source}:{name}: expected Zarr v3")
                if metadata.get("node_type") == "array" and not ARRAY_PATH.fullmatch(name):
                    raise ValueError(f"{source}:{name}: unsupported source array")
                dst.writestr(info.filename, data, compress_type=zipfile.ZIP_DEFLATED)
            elif "/c/" not in name:
                dst.writestr(info.filename, src.read(info), compress_type=zipfile.ZIP_DEFLATED)


def _read_sample(group, lead_names, cells, schema, definitions, path):
    values = {}
    for name, (_, rest) in definitions.items():
        dtype = np.dtype(bool) if name == "class_token_present" else schema[name][1]
        if dtype.kind == "M":
            dtype = np.dtype("datetime64[ns]")
        values[name] = np.zeros((len(lead_names), *rest), dtype=dtype)
    for index, lead in enumerate(lead_names):
        state = group["latent"][lead]
        arrays = _state_arrays(state, f"{path}/latent/{lead}")
        for name, array in arrays.items():
            if array is None:
                continue
            if array.shape != schema[name][0] or np.dtype(array.dtype) != schema[name][1]:
                raise ValueError(
                    f"{path}/latent/{lead}: inconsistent {name} shape or dtype "
                    f"{array.shape}, {array.dtype}; expected {schema[name]}"
                )
            if name == "tokens":
                data = array.oindex[cells, :, 0]
            elif name in ("prediction_coords", "prediction_geoinfo"):
                data = array.oindex[cells, :]
            elif name == "prediction_times":
                data = array.oindex[cells]
            else:
                data = array[:]
            if data.dtype.kind == "M":
                try:
                    converted = data.astype("datetime64[ns]")
                except OverflowError as error:
                    raise ValueError(
                        f"{path}/latent/{lead}: {name} cannot be represented in nanoseconds"
                    ) from error
                if not np.array_equal(converted.astype(data.dtype), data, equal_nan=True):
                    raise ValueError(
                        f"{path}/latent/{lead}: {name} cannot be represented in nanoseconds"
                    )
                data = converted
            values[name][index] = data
        values["class_token_present"][index] = arrays["class_token"] is not None
    return xr.Dataset(
        {
            name: (("sample", "lead", *definitions[name][0]), data[np.newaxis, ...])
            for name, data in values.items()
        }
    )


def _new_output(source, output):
    source, output = pathlib.Path(source).resolve(), pathlib.Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    return source, output


def _initialize(source, output, samples, first_run, bounds, years=None):
    cells = select_cells(**bounds)
    first_path, first_position, _, _, _ = samples[0]
    with ZipStore(first_path, mode="r") as store:
        root = zarr.open_group(store, mode="r")
        groups = _sample_groups(root, first_run[2]["inits"][first_path.name], first_path)
        group = root[groups[first_position]]
        lead_names, steps = _lead_steps(group, first_path)
        schema = _schema(group["latent"][lead_names[0]], f"{first_path}/{groups[first_position]}")
    ds, encoding = _dataset(samples, steps, cells, schema, bounds, source)
    if years is not None:
        ds.attrs["export_plan"] = {
            "years": years,
            "steps": steps,
            "schema": {name: (shape, dtype.str) for name, (shape, dtype) in schema.items()},
        }
    output.mkdir(parents=True, exist_ok=False)
    ds.to_zarr(
        output, mode="w", zarr_format=3, consolidated=False, compute=False, encoding=encoding
    )
    zarr.open_group(output, mode="a").create_group("provenance")
    LOG.info("Selected %d / %d cells across %d samples", len(cells), N_CELLS, len(samples))
    return cells, steps, schema


def _write_samples(output, runs, cells, steps, schema, start, stop):
    definitions = _definitions(schema, len(cells))
    sample_index = start
    for manifest_path, relative, manifest in runs:
        provenance = output / "provenance" / relative
        provenance.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, provenance / "MANIFEST.json")
        for name in manifest["files"]:
            if name not in manifest["inits"]:
                shutil.copyfile(manifest_path.parent / name, provenance / name)
        for name, bases in manifest["inits"].items():
            path = manifest_path.parent / name
            LOG.info("%s: %s", relative, name)
            _save_zip_metadata(path, provenance / f"{name}.metadata.zip")
            with ZipStore(path, mode="r") as store:
                root = zarr.open_group(store, mode="r")
                groups = _sample_groups(root, bases, path)
                for group_name in groups:
                    if sample_index >= stop:
                        raise ValueError(f"{path}: sample would exceed the assigned range")
                    group = root[group_name]
                    names, actual_steps = _lead_steps(group, f"{path}/{group_name}")
                    if actual_steps != steps:
                        raise ValueError(
                            f"{path}/{group_name}: inconsistent lead steps {actual_steps}"
                        )
                    batch = _read_sample(
                        group, names, cells, schema, definitions, f"{path}/{group_name}"
                    )
                    batch.to_zarr(
                        output,
                        mode="r+",
                        region={"sample": slice(sample_index, sample_index + 1)},
                        consolidated=False,
                    )
                    sample_index += 1
    if sample_index != stop:
        raise ValueError(f"wrote {sample_index - start} samples, expected {stop - start}")


def subset_archive(source, output, south, north, west, east):
    """Write all selected samples to a new regional Zarr dataset."""
    source, output = _new_output(source, output)
    runs, samples = _discover(source)
    first_run = next(run for run in runs if run[0].parent == samples[0][0].parent)
    bounds = {"south": south, "north": north, "west": west, "east": east}
    cells, steps, schema = _initialize(source, output, samples, first_run, bounds)
    _write_samples(output, runs, cells, steps, schema, 0, len(samples))
    zarr.open_group(output, mode="a").attrs["export_status"] = "complete"
    return cells


def _manifest_digest(runs):
    # Preserve manifest order because it determines the sample offsets.
    contents = [(str(relative), manifest) for _, relative, manifest in runs]
    return hashlib.sha256(json.dumps(contents).encode()).hexdigest()


def initialize_years(source, output, first_year, last_year, **bounds):
    source, output = _new_output(source, output)
    if not 1000 <= first_year <= last_year <= 9999:
        raise ValueError("expected an ascending range of four-digit years")
    samples, years = [], {}
    first_run = None
    for year in range(first_year, last_year + 1):
        runs, year_samples = _discover(source / str(year), source)
        if first_run is None:
            first_run = next(run for run in runs if run[0].parent == year_samples[0][0].parent)
        start = len(samples)
        samples.extend(year_samples)
        years[str(year)] = [start, len(samples), _manifest_digest(runs)]
    _initialize(source, output, samples, first_run, bounds, years)
    (output / "provenance" / ".export").mkdir()


def write_year(output, year):
    output = output.resolve()
    root = zarr.open_group(output, mode="r")
    plan = root.attrs.get("export_plan")
    year = str(year)
    if plan is None or year not in plan["years"]:
        raise ValueError(f"{output}: no export assignment for {year}")
    state = output / "provenance" / ".export"
    marker = state / f"{year}.complete"
    with (state / f"{year}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"{output}: another worker is writing {year}") from error
        if marker.is_file():
            LOG.info("%s is already complete", year)
            return
        start, stop, digest = plan["years"][year]
        source = pathlib.Path(root.attrs["source_directory"])
        runs, samples = _discover(source / year, source)
        if len(samples) != stop - start or _manifest_digest(runs) != digest:
            raise ValueError(f"{source / year}: source manifests changed since initialization")
        schema = {
            name: (tuple(shape), np.dtype(dtype)) for name, (shape, dtype) in plan["schema"].items()
        }
        LOG.info("Writing %s to sample range [%d, %d)", year, start, stop)
        _write_samples(output, runs, root["cell"][:], plan["steps"], schema, start, stop)
        marker.touch(exist_ok=False)


def finalize_years(output):
    root = zarr.open_group(output, mode="r+")
    plan = root.attrs.get("export_plan")
    if plan is None:
        raise ValueError(f"{output}: no yearly export plan")
    state = output / "provenance" / ".export"
    missing = [year for year in plan["years"] if not (state / f"{year}.complete").is_file()]
    if missing:
        raise ValueError(f"{output}: unfinished years: {', '.join(missing)}")
    root.attrs["export_status"] = "complete"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"init": initialize_years, "write-year": write_year, "finalize": finalize_years}
    stage = argv[0] if argv and argv[0] in stages else None
    if stage:
        parser.prog += f" {stage}"
        argv = argv[1:]
    else:
        parser.epilog = "Parallel export stages: init, write-year, finalize."
    if stage in (None, "init"):
        parser.add_argument("source", type=pathlib.Path, help="raw archive, year, or run directory")
        for bound in ("south", "north", "west", "east"):
            parser.add_argument(f"--{bound}", type=float, required=True)
    parser.add_argument("output", type=pathlib.Path, help="output Zarr directory")
    if stage == "init":
        parser.add_argument("first_year", type=int)
        parser.add_argument("last_year", type=int)
    elif stage == "write-year":
        parser.add_argument("year", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        stages.get(stage, subset_archive)(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
