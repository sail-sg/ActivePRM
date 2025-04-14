# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deep networks."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def init_weights(m):
    @torch.no_grad()
    def truncated_normal_init(t, mean=0.0, std=0.01):
        # torch.nn.init.normal_(t, mean=mean, std=std)
        t.data.normal_(mean, std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            w = torch.empty(t.shape, device=t.device, dtype=t.dtype)
            # torch.nn.init.normal_(w, mean=mean, std=std)
            w.data.normal_(mean, std)
            t = torch.where(cond, w, t)
        return t

    if type(m) is nn.Linear or isinstance(m, EnsembleFC):
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m.in_features)))
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def init_weights_uniform(m):
    input_dim = m.in_features
    torch.nn.init.uniform(m.weight, -1 / np.sqrt(input_dim), 1 / np.sqrt(input_dim))
    if m.bias is not None:
        m.bias.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, encoding_dim, hidden_dim=128, activation="relu") -> None:
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_dim
        self.output_dim = 1

        self.nn1 = nn.Linear(encoding_dim, hidden_dim)
        self.nn2 = nn.Linear(hidden_dim, hidden_dim)
        self.nn_out = nn.Linear(hidden_dim, self.output_dim)

        self.apply(init_weights)

        if activation == "swish":
            self.activation = Swish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.nn1(encoding))
        x = self.activation(self.nn2(x))
        score = self.nn_out(x)
        return score

    def init(self):
        self.init_params = self.get_params().data.clone()
        if torch.cuda.is_available():
            self.init_params = self.init_params.cuda()

    def regularization(self):
        """Prior towards independent initialization."""
        return ((self.get_params() - self.init_params) ** 2).mean()


class EnsembleFC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
        dtype=torch.float32,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        # init immediately to avoid error
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.weight.dtype)
        wx = torch.einsum("eblh,ehm->eblm", input, self.weight)

        return torch.add(wx, self.bias[:, None, None, :])  # w times x + b


def get_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


class _EnsembleModel(nn.Module):
    def __init__(self, encoding_dim, num_ensemble, hidden_dim=128, activation="relu", dtype=torch.float32) -> None:
        # super().__init__(encoding_dim, hidden_dim, activation)
        super(_EnsembleModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.hidden_dim = hidden_dim
        self.output_dim = 1

        self.nn1 = EnsembleFC(encoding_dim, hidden_dim, num_ensemble, dtype=dtype)
        self.nn2 = EnsembleFC(hidden_dim, hidden_dim, num_ensemble, dtype=dtype)
        self.nn_out = EnsembleFC(hidden_dim, self.output_dim, num_ensemble, dtype=dtype)

        self.apply(init_weights)

        if activation == "swish":
            self.activation = Swish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.nn1(encoding))
        x = self.activation(self.nn2(x))
        score = self.nn_out(x)
        return score

    def regularization(self):
        """Prior towards independent initialization."""
        return ((self.get_params() - self.init_params) ** 2).mean()


class EnsembleModel(nn.Module):
    def __init__(self, encoding_dim, num_ensemble, hidden_dim=128, activation="relu", dtype=torch.float32) -> None:
        super(EnsembleModel, self).__init__()
        self.encoding_dim = encoding_dim
        self.num_ensemble = num_ensemble
        self.hidden_dim = hidden_dim
        self.model = _EnsembleModel(encoding_dim, num_ensemble, hidden_dim, activation, dtype)
        self.reg_model = deepcopy(self.model)  # only used for regularization
        # freeze the reg model
        for param in self.reg_model.parameters():
            param.requires_grad = False

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.model(encoding)

    def regularization(self):
        """Prior towards independent initialization."""
        model_params = get_params(self.model)
        reg_params = get_params(self.reg_model).detach()
        return ((model_params - reg_params) ** 2).mean()
