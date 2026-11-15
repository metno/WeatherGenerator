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
`training_config.forecast.num_members` sets the number of parallel replicas, independently
of `training_config.noise.enabled`. Consecutive ranks share inputs and model weights;
enabled noise adds independent perturbations. The world size must be divisible by the
member count. Physical predictions are gathered with gradient communication for CRPS;
do not divide the loss by the member count again.

Validation inherits the training settings and count. Separate inference can override
`test_config.forecast.num_members` with a compatible rank count. The overlay also enables
`training_config.noise.condition_on_forecast_step`; enabling it on an existing unconditioned
injector changes its first MLP layer's input size and requires a weight migration or reset.
GPU/checkpoint validation must be done on the training server.

# Development guidelines

The [main branch](https://github.com/ecmwf/WeatherGenerator/tree/main) is the most stable version. If you are running experiments, you should use this branch.

The [develop branch](https://github.com/ecmwf/WeatherGenerator/tree/develop) has the latest
features. However, it is currently evolving at a fast pace. It should not be expected to have stable code or weight interfaces, or to be backward compatible.

# Copyright and License

This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
