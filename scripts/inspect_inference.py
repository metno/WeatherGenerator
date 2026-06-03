#!/usr/bin/env python
"""Quick script to inspect zarr file contents."""

import argparse
import sys

import numpy as np

from weathergen.common.io import zarrio_reader
from weathergen.common.config import get_model_results


def _unique_times(dataset):
    return np.unique(np.asarray(dataset.times).astype("datetime64[ns]"))


def _format_time_label(unique_times):
    if len(unique_times) == 0:
        return "none"
    if len(unique_times) == 1:
        return str(unique_times[0])
    return f"{unique_times[0]} -> {unique_times[-1]}"


def _print_time_info(label, dataset):
    unique_times = _unique_times(dataset)
    if len(unique_times) == 0:
        print(f"  {label} time steps: none")
        return

    print(f"  {label} first time step: {unique_times[0]}")
    if len(unique_times) > 1:
        print(f"  {label} last time step: {unique_times[-1]}")


def _representative_dataset(output_item):
    for name in ("target", "prediction", "source"):
        dataset = getattr(output_item, name)
        if dataset is not None:
            return dataset
    return None


def _selected_samples(samples, edge_count=2):
    samples_sorted = sorted(int(s) for s in samples)
    selected = samples_sorted[:edge_count] + samples_sorted[-edge_count:]
    # Keep order while removing duplicates (important when total samples < 2 * edge_count).
    return list(dict.fromkeys(selected))


def _format_samples_overview(samples, edge_count=2):
    samples_sorted = sorted(int(s) for s in samples)
    selected = _selected_samples(samples_sorted, edge_count=edge_count)
    if len(selected) == len(samples_sorted):
        return str(selected)

    first = ", ".join(str(s) for s in selected[:edge_count])
    last = ", ".join(str(s) for s in selected[-edge_count:])
    return f"[{first}, ..., {last}]"


def _collect_stream_sample_labels(zio, forecast_step, selected_samples):
    stream_labels = {}
    for stream in zio.streams:
        labels = []
        for sample in selected_samples:
            data = zio.get_data(sample, stream, forecast_step)
            dataset = _representative_dataset(data)
            if dataset is None:
                label = "no data"
            else:
                label = _format_time_label(_unique_times(dataset))

            labels.append((sample, label))

        stream_labels[stream] = labels

    return stream_labels


def _print_sample_time_table(zio, forecast_step):
    edge_count = 2
    samples_all = sorted(int(s) for s in zio.samples)
    selected_samples = _selected_samples(samples_all, edge_count=edge_count)
    stream_labels = _collect_stream_sample_labels(zio, forecast_step, selected_samples)
    print("\nSample -> timestamp by stream:")
    if len(samples_all) > len(selected_samples):
        skipped = len(samples_all) - len(selected_samples)
        print(f"  Showing {len(selected_samples)} edge samples (skipping {skipped} middle samples)")

    show_gap = len(samples_all) > len(selected_samples)

    label_sequences = [tuple(labels) for labels in stream_labels.values()]
    if label_sequences and all(labels == label_sequences[0] for labels in label_sequences[1:]):
        stream_names = ", ".join(stream_labels.keys())
        print(f"  Shared across streams: {stream_names}")
        labels = list(label_sequences[0])
        for sample, label in labels[:edge_count]:
            print(f"    sample {sample:>3}: {label}")
        if show_gap:
            print("    ...")
        for sample, label in labels[edge_count:]:
            print(f"    sample {sample:>3}: {label}")
        return

    for stream, labels in stream_labels.items():
        print(f"  {stream}:")
        for sample, label in labels[:edge_count]:
            print(f"    sample {sample:>3}: {label}")
        if show_gap:
            print("    ...")
        for sample, label in labels[edge_count:]:
            print(f"    sample {sample:>3}: {label}")


def _print_compact_time_count(zio, forecast_step):
    all_times = []
    for stream in zio.streams:
        for sample in sorted(int(s) for s in zio.samples):
            data = zio.get_data(sample, stream, forecast_step)
            dataset = _representative_dataset(data)
            if dataset is None:
                continue

            unique_times = _unique_times(dataset)
            if len(unique_times) > 0:
                all_times.append(unique_times)

    if len(all_times) == 0:
        print("Unique timestamps across samples: 0")
        print("Smallest timestamp: none")
        print("Largest timestamp: none")
        return

    unique_times = np.unique(np.concatenate(all_times))

    print(f"Unique timestamps across samples: {len(unique_times)}")
    print(f"Smallest timestamp: {unique_times[0]}")
    print(f"Largest timestamp: {unique_times[-1]}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("epoch", nargs="?", type=int, default=0)
    parser.add_argument("rank", nargs="?", type=int, default=0)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show only the total number of unique timestamps across samples.",
    )
    return parser.parse_args()


args = _parse_args()
run_id = args.run_id
epoch = args.epoch
rank = args.rank

print(f"Inspecting run: {run_id}, epoch: {epoch}, rank: {rank}")
print("=" * 60)

try:
    fname_zarr = get_model_results(run_id, epoch, rank)
    print(f"Zarr path: {fname_zarr}\n")
    
    with zarrio_reader(fname_zarr) as zio:
        samples_all = sorted(int(s) for s in zio.samples)
        fsteps_all = sorted(int(f) for f in zio.forecast_steps)
        print(f"Samples: {_format_samples_overview(samples_all)}")
        print(f"Streams: {list(zio.streams)}")
        print(f"Forecast steps: {fsteps_all}\n")
        
        # Get first available data
        sample = samples_all[0]
        stream = zio.streams[0]
        fstep = fsteps_all[0]
        
        print(f"Inspecting sample {sample}, stream {stream}, fstep {fstep}:")
        data = zio.get_data(sample, stream, fstep)
        
        print(f"  Target channels: {data.target.channels}")
        print(f"  Target shape: {data.target.data.shape}")
        _print_time_info("Target", data.target)
        print(f"  Prediction channels: {data.prediction.channels}")
        print(f"  Prediction shape: {data.prediction.data.shape}")
        _print_time_info("Prediction", data.prediction)
        if args.compact:
            print()
            _print_compact_time_count(zio, fstep)
        else:
            _print_sample_time_table(zio, fstep)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
