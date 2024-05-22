# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from typing import Dict, List

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
)
from secretflow.utils.simulation.datasets import load_mnist


class MnistBase(ApplicationBase, ABC):
    def __init__(
        self,
        alice,
        bob,
        has_custom_dataset=False,
        epoch=5,
        train_batch_size=128,
        hidden_size=612,
        dnn_fuse_units_size=None,
    ):
        super().__init__(
            alice,
            bob,
            device_y=bob,
            has_custom_dataset=has_custom_dataset,
            total_fea_nums=4000,
            alice_fea_nums=2000,
            num_classes=10,
            epoch=epoch,
            train_batch_size=train_batch_size,
            hidden_size=hidden_size,
            dnn_fuse_units_size=dnn_fuse_units_size,
        )
        self.train_dataset_len = 60000
        self.test_dataset_len = 10000
        if global_config.is_simple_test():
            self.train_dataset_len = 4000
            self.test_dataset_len = 4000

    def dataset_name(self):
        return 'mnist'

    def prepare_data(self, parts=None, is_torch=True, normalized_x=True):
        if parts is None:
            parts = {self.alice: (0, 14), self.bob: (14, 28)}
        (train_data, train_label), (test_data, test_label) = load_mnist(
            parts=parts,
            is_torch=is_torch,
            normalized_x=normalized_x,
            axis=3,
        )

        train_data = train_data.astype(np.float32)
        train_label = train_label
        test_data = test_data.astype(np.float32)
        test_label = test_label

        train_label.partitions.pop(self.alice)
        test_label.partitions.pop(self.alice)

        if global_config.is_simple_test():
            sample_nums = 4000
            train_data = train_data[0:sample_nums]
            train_label = train_label[0:sample_nums]
            test_data = test_data[0:sample_nums]
            test_label = test_label[0:sample_nums]

        return train_data, train_label, test_data, test_label

    def resources_consumes(self) -> List[Dict]:
        return [
            {'alice': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
            {'bob': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
        ]

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_MulticlassAccuracy": "max",
            "train_MulticlassPrecision": "max",
            "train_MulticlassAUROC": "max",
            "val_MulticlassAccuracy": "max",
            "val_MulticlassPrecision": "max",
            "val_MulticlassAUROC": "max",
        }

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.MULTICLASS

    def base_input_mode(self) -> InputMode:
        return InputMode.SINGLE

    def dataset_type(self) -> DatasetType:
        return DatasetType.IMAGE
