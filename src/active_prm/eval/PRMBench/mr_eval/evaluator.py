import argparse
import gc
import itertools
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader

from .models import get_model
from .tasks import get_task_functions, get_task_object
from .tasks.base_dataset.base_evaluation_dataset import (
    BaseEvalDataset, DataCollatorForSupervisedDataset)
from .utils.arguments import *
from .utils.log_utils import get_logger
from .utils.utils import *

logger = get_logger(__name__)


class MREvaluator:
    def __init__(self, model_args, task_args, script_args):
        self.config = script_args.config
        self.model_args = model_args
        self.task_args = task_args
        self.script_args = script_args
        self.tasks = self.task_args.task_name
        self.model = get_model(self.model_args.model)(**self.model_args.model_args)
        try:
            self.tokenizer = self.model.tokenizer
        except AttributeError:
            self.tokenizer = None

        self.state = AcceleratorState()
        self.batch_size = asdict(self.model_args).get("batch_size", 1)
        if self.state.deepspeed_plugin:
            deepspeed_config = self.state.deepspeed_plugin.deepspeed_config
            # 修改配置
            deepspeed_config["train_micro_batch_size_per_gpu"] = self.batch_size
            # 应用修改
            self.state.deepspeed_plugin.deepspeed_config = deepspeed_config
        else:
            logger.info("DeepSpeed is not initialized. Skipping DeepSpeed-specific configuration.")

    def evaluate(self):
        for task_name in self.tasks:
            logger.info(f"evaluating {task_name}")
            task_dict = get_task_functions(task_name)
            load_data_function, evaluate_function, task_config = (
                task_dict["load_data_function"],
                task_dict["evaluate_function"],
                task_dict["task_config"],
            )
            self.model.set_generation_config(task_config.generation_config)

            dataset = BaseEvalDataset(
                load_data_function=load_data_function,
                getitem_function=self.model.getitem_function,
                evaluate_function=evaluate_function,
                task_config=task_config,
                task_args=self.task_args,
                model_args=self.model_args,
            )
            num_workers = self.model_args.num_workers
            data_collator = DataCollatorForSupervisedDataset(
                tokenizer=self.tokenizer,
                max_length=task_config.generation_config.max_length,
                padding_side=dataset.padding_side,
            )
            dataloader = DataLoader(
                dataset, batch_size=self.model_args.batch_size, num_workers=num_workers, collate_fn=data_collator
            )
            self.model.respond(dataloader)
            res_log = dataset.evaluate()
            if is_main_process():
                logger.info(f"evaluation of {task_name} completed")
                append_jsonl(res_log, self.script_args.output_path)
