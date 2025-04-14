import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from datasets import (DatasetDict, concatenate_datasets, load_dataset,
                      load_from_disk)
from transformers import AutoTokenizer, TrainerCallback
from trl import (ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map,
                 get_quantization_config)

from active_prm.dataset import PRM800K, ProcessBenchAll
from active_prm.models import AutoModelForEnsemblePRM
from active_prm.models.utils import DataCollatorForSeq2Seq
from active_prm.trainer import ActiveSFTConfig, ActiveSFTTrainer

SYSTEM_PROMPT = "Please think step-by-step and put your final answer within \\boxed{}."


def get_pos(t):
    zero_positions = np.where(t == 0)[0]  # Get indices of 0s
    first_zero_pos = zero_positions[0] if zero_positions.size > 0 else -1
    return first_zero_pos


def eval_results(preds, labels, threshold):
    pred_poses, label_poses = [], []
    for pred, label in zip(preds, labels):
        pred = pred[label != -100]
        label = label[label != -100]
        pred_poses.append(get_pos(pred >= threshold))
        label_poses.append(get_pos(label))

    pred_poses, label_poses = np.array(pred_poses), np.array(label_poses)
    positive_mask = label_poses == -1
    _pos = np.mean(label_poses[positive_mask] == pred_poses[positive_mask])
    _neg = np.mean(label_poses[~positive_mask] == pred_poses[~positive_mask])
    # print(f"Positive: {_pos:.4f}, Negative: {_neg:.4f}")
    return _pos, _neg, 2 * _pos * _neg / (_pos + _neg)


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    data = {
        "math": (preds[:1000], labels[:1000]),
        "olympiadbench": (preds[1000:2000], labels[1000:2000]),
        "omnimath": (preds[2000:3000], labels[2000:3000]),
        "gsm8k": (preds[3000:], labels[3000:]),
    }
    # for pred, label in zip(preds, labels):
    thresholds = np.arange(0.0, 1.0, 0.1)
    # thresholds = [0.5]
    metrics = {}
    best_threshold = None
    for i, (subset, (_preds, _labels)) in enumerate(data.items()):
        if i == 0:
            f1_list, pos_list, neg_list = [], [], []
            for threshold in thresholds:
                pos, neg, f1 = eval_results(_preds, _labels, threshold)
                f1_list.append(f1)
                pos_list.append(pos)
                neg_list.append(neg)
            best_threshold = thresholds[np.argmax(f1_list)]
            metrics.update(
                {
                    f"{subset}_f1": np.max(f1_list),
                    f"{subset}_pos": np.max(pos_list),
                    f"{subset}_neg": np.max(neg_list),
                    f"{subset}_threshold": best_threshold,
                }
            )
        else:
            pos, neg, f1 = eval_results(_preds, _labels, best_threshold)
            metrics.update({f"{subset}_f1": f1, f"{subset}_pos": pos, f"{subset}_neg": neg})
    metrics.update({"average_f1": np.mean([metrics[f"{subset}_f1"] for subset in data.keys()])})
    return metrics


class SaveLogHistoryCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        log_history_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "log_history.jsonl")
        os.makedirs(os.path.dirname(log_history_path), exist_ok=True)

        # Save the log_history to a JSON file
        df = pd.DataFrame(state.log_history)
        df.to_json(log_history_path, orient="records", lines=True)

        print(f"Saved log_history to {log_history_path}")
        return control


def main(script_args, training_args, model_args, custom_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    # automatically set the problem type
    custom_args.problem_type = "single_label_classification"
    if custom_args.label_type == "soft_labels" and custom_args.prm_type == "advantage":
        custom_args.problem_type = "regression"  # regression for advantage, containing negative values
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        num_ensemble=custom_args.num_ensemble,
        problem_type=custom_args.problem_type,
        learning_probability=custom_args.learning_probability,
        regularization_lambda=custom_args.regularization_lambda,
        rr_token=custom_args.rr_token,
    )
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Model
    ################
    model = AutoModelForEnsemblePRM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.eos_token_id
    # check whether rr_token is in tokenizer vocab; if not add it
    if custom_args.rr_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [custom_args.rr_token]})
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
        print(f"Added rr_token {custom_args.rr_token} to vocab.")
        # set the embedding of new token to the same as '\n\n'
        newline_token_id = tokenizer("\n\n").input_ids[0]
        newline_embedding = model.get_input_embeddings().weight[newline_token_id].detach()
        rr_token_id = tokenizer(custom_args.rr_token).input_ids[0]
        with torch.no_grad():
            model.get_input_embeddings().weight[rr_token_id] = newline_embedding
            print(f"Set the embedding of rr_token {custom_args.rr_token} to be the same as '\\n\\n'")
    else:
        print(f"rr_token {custom_args.rr_token} is already in vocab. No need to add it.")
    # rr_token_id
    model.config.rr_token_id = tokenizer(custom_args.rr_token).input_ids[0]

    if custom_args.freeze_backbone:
        for param in model.model.parameters():
            param.requires_grad_(False)
        print("Backbone weights are freezed. Perform finetuning on the head only!!!")
    else:
        print("Backbone weights are not freezed. Perform full finetuning!!!")

    ################
    # Dataset
    ################

    if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=script_args.dataset_name)
    elif script_args.dataset_name == "PRM800K":
        dataset = DatasetDict({"train": PRM800K().data})
    else:
        try:
            dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        except:
            dataset = load_from_disk(script_args.dataset_name)
            if not isinstance(dataset, DatasetDict):
                dataset = DatasetDict({"train": dataset})

    if custom_args.train_data_start_idx is not None and custom_args.train_data_end_idx is not None:
        start_idx = custom_args.train_data_start_idx
        end_idx = min(custom_args.train_data_end_idx, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(start_idx, end_idx))
        print(f"Selected data from {start_idx} to {end_idx}")

    ################
    # For dataset with for completion only sft
    ################

    def format_and_tokenize(example, label_type="hard_labels"):
        def _tokenize(text):
            return tokenizer(
                text, return_tensors="pt", padding=False, truncation=True, max_length=training_args.max_seq_length
            )

        def _strip(text_list):
            return [text.strip().replace("\n\n", "") for text in text_list]

        def _get_num_rr_in_question(question):
            id = tokenizer(rr_token).input_ids[0]
            return (_tokenize(question).input_ids == id).sum()

        rr_token = custom_args.rr_token
        target_labels = torch.tensor(example[label_type])
        len_target_labels = len(example[label_type])
        solution = rr_token.join(_strip(example["steps"][:len_target_labels])) + rr_token

        message = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": solution},
        ]
        formatted_text = tokenizer.apply_chat_template(message, tokenize=False)
        inputs = _tokenize(formatted_text)

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        # set labels only to rr_token positions

        labels = input_ids.clone().to(target_labels.dtype)
        labels[labels != tokenizer(rr_token).input_ids[0]] = -100

        # considering cases that question contains ' \n\n'
        num_rr_in_q = _get_num_rr_in_question(example["question"])
        # set the first num_rr_in_q True in labels to be False
        true_indices = torch.nonzero(labels != -100, as_tuple=True)[0]
        labels[true_indices[:num_rr_in_q]] = -100

        try:
            labels[labels != -100] = target_labels
        except:  # possibly caused by truncation
            _len = (labels != -100).sum()
            labels[labels != -100] = target_labels[:_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(
        format_and_tokenize,
        fn_kwargs={"label_type": custom_args.label_type},
        remove_columns=dataset["train"].features,
        batched=False,
        keep_in_memory=True,  # for saving cache disk memory
        num_proc=4,
    )

    if training_args.eval_strategy != "no":
        print("Loading and processing evaluation dataset processbench")
        eval_dataset = ProcessBenchAll().data
        eval_dataset = eval_dataset.map(
            format_and_tokenize,
            fn_kwargs={"label_type": custom_args.label_type},
            remove_columns=eval_dataset["gsm8k"].features,
            batched=False,
            keep_in_memory=True,
            num_proc=4,
        )
        # concatanate the eval_dataset subsets
        eval_dataset = concatenate_datasets(
            [eval_dataset[subset] for subset in ["math", "olympiadbench", "omnimath", "gsm8k"]]
        )
    ################
    # Training
    ################
    trainer = ActiveSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
        compute_metrics=compute_metrics,
        callbacks=[SaveLogHistoryCallback()],
    )

    latest_checkpoint = None
    if os.path.exists(training_args.output_dir):
        checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = os.path.join(
                training_args.output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1]))
            )

    if latest_checkpoint is not None:
        print(f"Resuming training from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    # Save and push to hub

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    if trainer.is_world_process_zero():
        log_history = pd.DataFrame(trainer.state.log_history)
        log_history.to_json(os.path.join(training_args.output_dir, "log_history.jsonl"), orient="records", lines=True)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # remove checkpoint* subdirs
    os.system(f"rm -rf {training_args.output_dir}/checkpoint*")


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, ActiveSFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    parser.add_argument("--label_type", type=str, default="soft_labels", choices=["soft_labels", "hard_labels"])
    parser.add_argument("--prm_type", type=str, default="value", choices=["value", "advantage"])
    parser.add_argument("--normalization_type", type=str, default="no_norm", choices=["norm", "no_norm"])
    parser.add_argument("--train_data_start_idx", type=int, default=None)
    parser.add_argument("--train_data_end_idx", type=int, default=None)
    parser.add_argument("--freeze_backbone", type=bool, default=False)
    # for ensemble prm
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--problem_type", type=str, default="single_label_classification")
    parser.add_argument("--learning_probability", type=float, default=1.0)
    parser.add_argument("--regularization_lambda", type=float, default=0.5)
    parser.add_argument("--rr_token", type=str, default="<extra_0>")
    parser.add_argument(
        "--trainer_class", type=str, default="active_v3", choices=["active_v3", "random_selection", "sft"]
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, custom_args)
