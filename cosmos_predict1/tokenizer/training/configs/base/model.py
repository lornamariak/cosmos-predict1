# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import attrs

from cosmos_predict1.tokenizer.training.model import TokenizerModel
from cosmos_predict1.utils.config import EMAConfig
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class ModelConfig:
    network: LazyDict = None
    loss: LazyDict = None
    metric: LazyDict = None
    ema: EMAConfig = EMAConfig(enabled=True, beta=0.9999)
    precision: str = "bfloat16"
    torch_compile: bool = False
    disc: LazyDict = None
    disc_optimizer: LazyDict = None
    disc_scheduler: LazyDict = None


DefaultModelConfig: LazyDict = L(TokenizerModel)(config=ModelConfig())
