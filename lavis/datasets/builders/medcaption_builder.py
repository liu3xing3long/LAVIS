"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.mimic_caption_datasets import (
    MIMICCapDataset,
    MIMICCapEvalDataset,
    NoCapsEvalDataset,
)
from lavis.datasets.datasets.medicat_caption_datasets import (
    MedICaTCapDataset,
    MedICaTCapEvalDataset,
    NoCapsEvalDataset,
)
from lavis.datasets.datasets.roco_caption_datasets import (
    ROCOCapDataset,
    ROCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("mimic_caption")
class MIMICCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = MIMICCapDataset
    eval_dataset_cls = MIMICCapDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_cap.yaml",
    }
@registry.register_builder("mimic_nocaps")
class MIMICCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_nocaps.yaml",
    }

@registry.register_builder("medicat_caption")
class MedICaTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = MedICaTCapDataset
    eval_dataset_cls = MedICaTCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_cap.yaml",
    }
@registry.register_builder("medicat_nocaps")
class MedICaTCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_nocaps.yaml",
    }

@registry.register_builder("roco_caption")
class ROCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ROCOCapDataset
    eval_dataset_cls = ROCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_cap.yaml",
    }
@registry.register_builder("roco_nocaps")
class ROCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/med/defaults_nocaps.yaml",
    }
