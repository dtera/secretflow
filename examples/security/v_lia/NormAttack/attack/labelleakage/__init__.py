# Copyright 2024 Ant Group Co., Ltd.
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

"""Subpackage for label leakage attack, which infere the private label information of the training dataset.
"""

from .normattack import NormAttackSplitNNManager  # noqa: F401
from .normattack import (
    NormAttackSplitNNManager_sf,
    attach_normattack_to_splitnn,
    attach_normattack_to_splitnn_sf,
)
