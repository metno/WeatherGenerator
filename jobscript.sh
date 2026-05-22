#!/bin/bash

#SBATCH --partition=gpuB-research 
#SBATCH --account=bris-dev 
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=140G
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/test.out

ulimit -v unlimited 2>/dev/null || echo "Cannot set virtual memory limit"
export PYTHONPYCACHEPREFIX="/tmp/pycache_${USER}"

START_DATE=2020-01-01T00:00
END_DATE=2020-12-31T18:00
NUM_STEPS=40  # Forecast length, times 6h

uv run --offline inference \
	--base-config=config/config_pre-ops.yml \
	-id=atmofs03 \
	--options \
	    training_config.forecast.num_steps=${NUM_STEPS} \
	    test_config.start_date=${START_DATE} \
	    test_config.end_date=${END_DATE} \
	    test_config.samples_per_mini_epoch=1 \
	    test_config.output.num_samples=1 \
	    test_config.output.streams=[ERA5,latent] \
	    inference_only=True
