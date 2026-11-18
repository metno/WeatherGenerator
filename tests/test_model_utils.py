import torch
from torch import nn
from torch.nn import functional as functional

from weathergen.model.utils import disable_dropout


class AttentionDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_rate = 0.5

    def forward(self, inputs):
        return functional.scaled_dot_product_attention(
            inputs,
            inputs,
            inputs,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )


def test_disable_dropout_survives_training_mode_changes():
    model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), AttentionDropout())
    state_keys = list(model.state_dict())
    model.apply(disable_dropout)
    inputs = torch.randn(2, 1, 16, 8)

    for training in (True, False, True):
        model.train(training)
        output = model(inputs)
        torch.testing.assert_close(output, model(inputs), rtol=0, atol=0)
        assert all(module.training == training for module in model.modules())

    output.sum().backward()
    assert model[0].weight.grad is not None
    assert model[1].p == 0.0
    assert model[2].dropout_rate == 0.0
    assert list(model.state_dict()) == state_keys


def test_disable_dropout_preserves_other_randomness():
    class StochasticModule(nn.Module):
        def forward(self, inputs):
            return inputs + torch.randn_like(inputs)

    model = nn.Sequential(nn.Dropout(0.5), StochasticModule())
    model.apply(disable_dropout)
    model.train()
    inputs = torch.ones(128)

    assert not torch.equal(model(inputs), model(inputs))
    assert model[1].training