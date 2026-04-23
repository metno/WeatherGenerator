#!/usr/bin/env python3
"""Quick script to inspect zarr file contents."""

import sys
from weathergen.common.io import zarrio_reader
from weathergen.common.config import get_model_results

if len(sys.argv) < 2:
    print("Usage: python inspect_zarr.py <run_id> [epoch] [rank]")
    sys.exit(1)

run_id = sys.argv[1]
epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0

print(f"Inspecting run: {run_id}, epoch: {epoch}, rank: {rank}")
print("=" * 60)

try:
    fname_zarr = get_model_results(run_id, epoch, rank)
    print(f"Zarr path: {fname_zarr}\n")
    
    with zarrio_reader(fname_zarr) as zio:
        print(f"Samples: {sorted([int(s) for s in zio.samples])}")
        print(f"Streams: {list(zio.streams)}")
        print(f"Forecast steps: {sorted([int(f) for f in zio.forecast_steps])}\n")
        
        # Get first available data
        sample = int(zio.samples[0])
        stream = zio.streams[0]
        fstep = int(zio.forecast_steps[0])
        
        print(f"Inspecting sample {sample}, stream {stream}, fstep {fstep}:")
        data = zio.get_data(sample, stream, fstep)
        
        print(f"  Target channels: {data.target.channels}")
        print(f"  Target shape: {data.target.data.shape}")
        print(f"  Prediction channels: {data.prediction.channels}")
        print(f"  Prediction shape: {data.prediction.data.shape}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
