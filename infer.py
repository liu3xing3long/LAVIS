"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import json
import logging


def init_logging(log_name):
    """
    Init for logging
    """
    logging.basicConfig(level=logging.DEBUG, format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', filename=log_name, filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def datestr():
    import time
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}{:02}_{}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min,
                                                    now.tm_sec, int(round(time.time() * 1000)))

def _log_msg(str_msg):
    logging.info(str_msg)


def main():
    outputpath = './testoutput/'
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)

    init_logging(fr'{outputpath}/{datestr()}.log')

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # load ckpts
    ckptpath_list = [
        # 200 words
        # '/mnt/lustrenew/liuxinglong/work/LAVIS/lavis/output/BLIP2/Instruct_BLIP/20231012064/checkpoint_8.pth',
        # 100-words
        # '/mnt/lustrenew/liuxinglong/work/LAVIS/lavis/output/BLIP2/Instruct_BLIP/20231013114/checkpoint_9.pth',
        # 50-words
        '/mnt/lustrenew/liuxinglong/work/LAVIS/lavis/output/BLIP2/Instruct_BLIP/20231017093/checkpoint_9.pth',
    ]

    _log_msg(fr'loading model')
    # loads InstructBLIP model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type="infer_vicuna7b",
        is_eval=True,
        # device=device,
        device='cpu',
    )
    _log_msg(fr'done')

    for ckptpath in ckptpath_list:
        _log_msg(fr'loading checkpoint from {ckptpath}')
        model.load_checkpoint(ckptpath)
        model.to(device)
        model.eval()
        _log_msg(fr'done')

        # load sample image
        imageroot = '/mnt/lustrenew/liuxinglong/xray_data/MedICaT/medicat_release/release/figures/'
        with open('./data/anno/medicat_test.json', 'rb') as fp:
            dat = json.load(fp)

            for ii, dd in enumerate(dat):
                if ii >= 10:
                    break

                _log_msg('->' * 30)
                golden_report = dd['caption']
                image_name = dd["image"]
                imagepath = fr'{imageroot}/{image_name}'

                _log_msg(fr'loading image from {imagepath}')
                raw_image = Image.open(imagepath).convert("RGB")
                _log_msg(fr'done')

                _log_msg('process vis embedding')
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                _log_msg('done')

                _log_msg('generating')
                prompt = "Question: describe the image. Answer:"
                answer = model.generate({"image": image, "prompt": prompt})
                _log_msg('done')

                _log_msg('-' * 15)
                _log_msg(fr'image {image_name}')
                _log_msg(fr'goldenrepot {golden_report}')
                _log_msg(fr'realreport {answer}')
                _log_msg('-' * 15)
                _log_msg('<-' * 30)


if __name__ == '__main__':
    main()

