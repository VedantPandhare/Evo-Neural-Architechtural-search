"""Genome -> PyTorch nn.Module builder."""

import torch
import torch.nn as nn

from genomes.operators import genome_to_build_spec, BuildSpec
from genomes.schema import validate_genome

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


class EvoNASModel(nn.Module):
    def __init__(self, build_spec: BuildSpec):
        super().__init__()
        self.build_spec = build_spec
        self.feature_layers = nn.ModuleList()
        self.classifier_layers = nn.ModuleList()
        self._flatten_index = None
        self._build()

    def _build(self):
        in_channels = self.build_spec.input_shape[0]
        in_features = None
        past_flatten = False

        for layer in self.build_spec.layers:
            ltype = layer["type"]
            if ltype == "conv2d":
                pad = layer["kernel_size"] // 2 if layer["padding"] == "same" else 0
                conv = nn.Conv2d(in_channels, layer["filters"], layer["kernel_size"],
                                  stride=layer["stride"], padding=pad)
                self.feature_layers.append(conv)
                if layer["batchnorm"]:
                    self.feature_layers.append(nn.BatchNorm2d(layer["filters"]))
                self.feature_layers.append(ACTIVATION_MAP[layer["activation"]]())
                in_channels = layer["filters"]
            elif ltype == "maxpool":
                self.feature_layers.append(nn.MaxPool2d(layer["kernel_size"], stride=layer["stride"]))
            elif ltype == "avgpool":
                self.feature_layers.append(nn.AvgPool2d(layer["kernel_size"], stride=layer["stride"]))
            elif ltype == "flatten":
                past_flatten = True
                in_features = self._infer_flatten_dim()
            elif ltype == "dropout":
                self.classifier_layers.append(nn.Dropout(layer["rate"]))
            elif ltype == "dense":
                if in_features is None:
                    # no conv/flatten preceded this: input is already flat (e.g. dense-only genome)
                    in_features = int(torch.tensor(self.build_spec.input_shape).prod().item())
                dense = nn.Linear(in_features, layer["units"])
                self.classifier_layers.append(dense)
                if layer["batchnorm"]:
                    self.classifier_layers.append(nn.BatchNorm1d(layer["units"]))
                self.classifier_layers.append(ACTIVATION_MAP[layer["activation"]]())
                in_features = layer["units"]
            else:
                raise ValueError(f"unknown layer type: {ltype}")

    def _infer_flatten_dim(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(2, *self.build_spec.input_shape)
            for layer in self.feature_layers:
                x = layer(x)
            flat_dim = int(x.numel() // x.shape[0])
        self.train(was_training)
        return flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feature_layers:
            x = layer(x)
        x = x.flatten(1)
        for layer in self.classifier_layers:
            x = layer(x)
        return x


def build_model(genome: dict) -> EvoNASModel:
    result = validate_genome(genome)
    if not result.valid:
        raise ValueError(f"invalid genome: {result.reason}")
    build_spec = genome_to_build_spec(genome)
    return EvoNASModel(build_spec)


def make_optimizer(model: nn.Module, genome: dict) -> torch.optim.Optimizer:
    opt_cls = OPTIMIZER_MAP[genome["optimizer"]]
    return opt_cls(model.parameters(), lr=genome["learning_rate"])
