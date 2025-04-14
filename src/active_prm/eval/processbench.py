import os
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset

from active_prm.utils.worker import (EnsemblePRMWorker, LLMasJudgerWorker,
                                     QwenPRMWorker)


class ProcessBench:
    def __init__(self, subset):
        self.data = load_dataset("Qwen/processbench")[subset]

    def process_results(self, preds):
        labels = np.array(self.data["label"])
        preds = np.array(preds)
        positive_mask = labels != -1
        _pos = np.mean(labels[positive_mask] == preds[positive_mask])
        _neg = np.mean(labels[~positive_mask] == preds[~positive_mask])
        # print(f"Positive: {_pos:.4f}, Negative: {_neg:.4f}")
        return _pos, _neg, 2 * _pos * _neg / (_pos + _neg)

    def process_data(self):
        def _process(example):
            list_strip = lambda x_list: [x.strip() for x in x_list]
            # return {"prompt": [example["problem"], list_strip(example["steps"])]}
            return {"prompt": [example["problem"], list_strip(example["steps"])]}

        return self.data.map(_process)


def predict_with_threshold(rating, threshold):
    rating = np.array(rating)
    preds = rating < threshold
    pos = np.argmax(preds)
    if preds.sum() == 0:
        pos = -1
    return pos


def predict_with_advantages(rating, threshold):
    rating = np.array(rating)
    advs_preds = (rating[1:] - rating[:-1]) < 0
    values_preds = rating[1:] < threshold
    preds = advs_preds & values_preds
    pos = np.argmax(preds)
    if preds.sum() == 0:
        pos = -1
    return pos


def judger_entrypoint(reward_model_path, subsets="math,gsm8k,olympiadbench,omnimath", n=1, temperature=0.0):
    def _get_list(k):
        if isinstance(k, Tuple):
            k_list = list(k)
        elif isinstance(k, str):
            k_list = str(k).split(",")
        return k_list

    def _process_data(example):
        list_strip = lambda x_list: [x.strip() for x in x_list]
        return [example["problem"], list_strip(example["steps"])]

    reasoning_parser = {}
    if "qwq" in reward_model_path.lower() or "r1" in reward_model_path.lower():
        reasoning_parser = {"reasoning_parser": "deepseek-r1"}

    worker = LLMasJudgerWorker(reward_model_path, **reasoning_parser)

    subsets = _get_list(subsets)
    assert all(subset in ["gsm8k", "math", "olympiadbench", "omnimath"] for subset in subsets)
    for i, subset in enumerate(subsets):
        process_bench = ProcessBench(subset)
        # process_bench.data = process_bench.data.shuffle().select(range(3))
        prompts = [_process_data(row) for row in process_bench.data]
        preds, total_outputs = worker.generate(
            prompts,
            verbose=True,
            max_new_tokens=8192,
            temperature=temperature,
            return_preds_only=False,
        )
        # preds = [predict_with_threshold(logit, 0) for logit in logits]
        pos, neg, f1 = process_bench.process_results(preds)
        print(f"Positive: {pos:.4f}, Negative: {neg:.4f}, F1: {f1:.4f} for {subset}")
        data = process_bench.data.to_pandas()
        data["labeling_outputs"] = total_outputs
        data["preds"] = preds
        model_id = reward_model_path.split("/")[-1]
        output_dir = f"./out/bench/processbench/{model_id}/{subset}"
        os.makedirs(output_dir, exist_ok=True)
        data.to_json(os.path.join(output_dir, "outputs.jsonl"), orient="records", lines=True)
        print(f"Outputs saved to {output_dir}")


def prm_entrypoint(reward_model_path, subsets="gsm8k,math,olympiadbench,omnimath", rating_threshold=None, **kwargs):
    def _get_list(k):
        if isinstance(k, Tuple):
            k_list = list(k)
        elif isinstance(k, str):
            k_list = str(k).split(",")
        elif isinstance(k, float):
            k_list = [k]
        return k_list

    def _process_data(example):
        list_strip = lambda x_list: [x.strip() for x in x_list]
        return [example["problem"], list_strip(example["steps"])]

    if reward_model_path in ["Qwen/Qwen2.5-Math-PRM-7B", "Qwen/Qwen2.5-Math-PRM-72B"]:
        worker = QwenPRMWorker(reward_model_path, torch_dtype=torch.bfloat16, **kwargs)
        worker.model.to("cuda")
    else:
        worker = EnsemblePRMWorker(reward_model_path, torch_dtype="auto", **kwargs)
        worker.model.config.problem_type = "single_label_classification"
        worker.model.to("cuda")

    subsets = _get_list(subsets)
    if rating_threshold is not None:
        thresholds = _get_list(rating_threshold)
    else:
        thresholds = [0.5]

    # TODO: THE FOLLOWING CODE IS NOT WORKING FOR JUDER
    best_threshold = None
    assert all(subset in ["gsm8k", "math", "olympiadbench", "omnimath"] for subset in subsets)
    average_f1 = []
    for i, subset in enumerate(subsets):
        process_bench = ProcessBench(subset)
        prompts = [_process_data(row) for row in process_bench.data]
        logits = worker.generate(prompts, batch_size=16, verbose=True)
        if i == 0:
            f1_list, pos_list, neg_list = [], [], []
            for threshold in thresholds:
                preds = [predict_with_threshold(logit, float(threshold)) for logit in logits]
                pos, neg, f1 = process_bench.process_results(preds)
                # print(f"F1 score: {acc:.4f} for {subset}")
                f1_list.append(f1)
                pos_list.append(pos)
                neg_list.append(neg)
            best_threshold = thresholds[np.argmax(f1_list)]
            print(f"Best F1: {max(f1_list):.4f} for {subset}, use this one {best_threshold} as the best threshold")
            print(
                f"Positive: {pos_list[np.argmax(f1_list)]:.4f}, Negative: {neg_list[np.argmax(f1_list)]:.4f}, F1: {f1_list[np.argmax(f1_list)]:.4f} for {subset}"
            )
            average_f1.append(f1_list[np.argmax(f1_list)])
        else:
            preds = [predict_with_threshold(logit, best_threshold) for logit in logits]
            pos, neg, f1 = process_bench.process_results(preds)
            print(f"Positive: {pos:.4f}, Negative: {neg:.4f}, F1: {f1:.4f} for {subset}")
            average_f1.append(f1)
    print(f"Average F1: {np.mean(average_f1):.4f}")


if __name__ == "__main__":
    import fire

    fire.Fire({"judger": judger_entrypoint, "prm": prm_entrypoint})
