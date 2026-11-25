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

# Development guidelines

The [main branch](https://github.com/ecmwf/WeatherGenerator/tree/main) is the most stable version. If you are running experiments, you should use this branch.

The [develop branch](https://github.com/ecmwf/WeatherGenerator/tree/develop) has the latest
features. However, it is currently evolving at a fast pace. It should not be expected to have stable code or weight interfaces, or to be backward compatible.

# Optimizers

AdamW remains the default, including its existing batch-size scaling of betas
and epsilon. Select AdEMAMix explicitly in your training configuration:

```yaml
training_config:
  optimizer:
    name: "ademamix"
    grad_clip: 1.0
    weight_decay: 0.1
    ademamix:
      betas: [0.9, 0.95, 0.9999]
      alpha: 8.0
      beta3_warmup: 100000
      alpha_warmup: 100000
      eps: 1.0e-7
  learning_rate_scheduling:
    lr_start: 1.0e-6
    lr_max: 6.4e-4
    lr_final_decay: 1.0e-6
    lr_final: 0.0
    num_steps_warmup: 512
    num_steps_cooldown: 512
    policy_warmup: "cosine"
    policy_decay: "constant"
    policy_cooldown: "linear"
    parallel_scaling_policy: "const"
```

This ports the optimizer settings used by
[Anemoi's AdEMAMix](https://github.com/ecmwf/anemoi-core/blob/main/training/src/anemoi/training/optimizers/AdEMAMix.py).
The algorithm is described in [the AdEMAMix paper](https://arxiv.org/abs/2409.03137).
The effective peak learning rate is explicitly `6.4e-4`: `4.0e-5` multiplied
by the supplied Anemoi data-parallel factor of 16. `"const"` disables WG's
additional parallel LR scaling; it does not disable the LR schedule.
The LR warmup/cooldown above are WG settings, not a reconstruction of the
unspecified Anemoi LR schedule.

AdEMAMix betas and epsilon are used unchanged, and cosine LR scheduling does
not cycle beta1. Its alpha warmup is linear; beta3 warmup is linear in EMA
half-life, from beta1 to beta3. Both count successful updates per parameter,
not samples, ensemble members, epochs, or LR-scheduler steps. Check the total
number of updates: runs shorter than 100,000 updates will not finish these
warmups. No-gradient parameters and updates skipped by gradient scaling do
not advance their optimizer counters.

AdEMAMix checkpoints include a matching `.optim` sidecar containing all three
EMA buffers and their counters, restored by `train_continue`, including sharded
parameters. A missing optimizer checkpoint raises an error rather than silently
resetting the warmups. Use `load_chkpt` for a weights-only start (including an
AdamW-trained model), with fresh AdEMAMix state. AdamW checkpoint behavior is
unchanged.

AdEMAMix uses three full-sized EMA buffers instead of AdamW's two (about 50%
more optimizer-buffer memory), and this implementation is not fused.

# Copyright and License

This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
