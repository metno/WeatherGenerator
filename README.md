<p align="center">
  <img src="assets/weathergenerator_logo.png" alt="WeatherGenerator" width="400px">
</p>

<div align="center">
  <h1>The WeatherGenerator <br> Machine Learning Earth System Model</h1>
</div>

The WeatherGenerator project is developing a machine learning-based Earth system model. 
It will be trained on a wide range of datasets, including reanalyses, forecast data and observations, to provide a robust and versatile model for the dynamics.
Through this, it can be used for a wide-range of applications. General updates are shared on the project website: [weathergenerator.eu](https://weathergenerator.eu/) 

More details coming soon. Please open an issue if you are interested in using the model.

<hr>

<p align="center">
  <img src="assets/weathergenerator_partner.png" alt="Partners" width="1000px">
</p>

# How to use the WeatherGenerator project

The model is currently being developed by the WeatherGenerator Consortium. If you want to
engage, you are encouraged to contact us first by opening an issue on Github.

## Member-parallel CRPS fine-tuning

On the training server, with the project environment, datasets and checkpoint `sdnbfpve`
available, launch two members on two GPUs:

```bash
torchrun --nproc-per-node=2 --module weathergen.run_train \
  train_continue --from-run-id sdnbfpve --mini-epoch -1 \
  --config config/config_noise_injection_crps_finetuning.yml
```

The overlay requests eight mini-epochs, ERA5 as an output-only stream and physical-space
almost-fair CRPS with `alpha=0.95`. Other stream settings are inherited from the saved run.
Consecutive ranks form a member group and consume the same examples with independent
latent noise. Four ranks form two groups, each with two members. The world size must be
divisible by the ensemble size. Shared model weights remain synchronized over all ranks;
do not divide the CRPS loss by the ensemble size again.

Set `noise_injector.member_parallel: false` for the original single-process, locally batched
ensemble. Distributed regression checks can be run on the server with
`PYTHONPATH=src pytest tests/test_member_parallel.py -q`. These checks use CPU/Gloo and
lightweight attention substitutes; the real checkpoint, CUDA/NCCL and FSDP training paths
still require a server smoke run before a full experiment.

# Development guidelines

The [main branch](https://github.com/ecmwf/WeatherGenerator/tree/main) is the most stable version. If you are running experiments, you should use this branch.

The [develop branch](https://github.com/ecmwf/WeatherGenerator/tree/develop) has the latest
features. However, it is currently evolving at a fast pace. It should not be expected to have stable code or weight interfaces, or to be backward compatible.

# Copyright and License

This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
