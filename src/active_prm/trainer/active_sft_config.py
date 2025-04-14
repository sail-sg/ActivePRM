# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class ActiveSFTConfig(SFTConfig):
    # parameters for active learning
    active_learning_pred_threshold: float = field(
        default=0.9, metadata={"help": "Prediction threshold for active learning"}
    )
    active_learning_std_threshold: float = field(
        default=0.01,
        metadata={"help": "Std threshold for active learning"},
    )
    active_learning_warmup_steps: int = field(default=-1, metadata={"help": "Warmup steps for active learning"})
    random_selection_threshold: float = field(
        default=0.5, metadata={"help": "Random selection threshold as a baseline"}
    )
