import os
import re
import shutil
import uuid
from os.path import exists

import numpy as np
from datasets import (Dataset, load_dataset, load_dataset_builder,
                      load_from_disk)
from langdetect import detect
from math_verify import parse, verify

# from .utils.math import is_equiv, rex


def _is_equiv(results, answers):
    return [verify(answer, result) for answer, result in zip(answers, results)]


def _rex(results):
    return [parse(result) for result in results]


class DatasetBase:
    def __init__(self, repo, split, force_reprocess=False, subset=None, processed_name="processed"):
        # get the cache_dir first without load dataset
        self.ds_builder = load_dataset_builder(repo, subset)
        # cache_dir = os.path.dirname(self.data.cache_files[0]["filename"])
        cache_dir = self.ds_builder._cache_dir
        cache_dir = os.path.join(cache_dir, processed_name)
        if os.path.exists(cache_dir) and not force_reprocess:
            # should check whether it will create new cache file every time
            self.data = load_from_disk(cache_dir)
        else:
            # first load data
            self.data = load_dataset(repo, subset)[split]
            # remove all files in cache_dir if exists
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            self.data = self.process_data()
            self.save_cache(cache_dir)

    def process_data(self):
        pass

    def accuracy(self, results):
        results = _rex(results)
        return _is_equiv(results, self.answer)

    def accuracy_self_consistency(self, results):
        assert type(results[0]) is list
        maj = []
        for result_per_q in results:
            rex_results_per_q = _rex(result_per_q)
            maj.append(max(set(rex_results_per_q), key=rex_results_per_q.count))
        acc = _is_equiv(maj, self.answer)
        return acc

    def accuracy_upper_bound(self, results):
        assert type(results[0]) is list
        # transpose results
        results = list(zip(*results))
        results = [list(tup) for tup in results]

        results = [
            # [rex(answer, result) for answer, result in zip(self.answer, result_per_trial)]
            _is_equiv(_rex(result_per_trial), self.answer)
            for result_per_trial in results
        ]
        results = np.array(results).any(axis=0)
        return results.tolist()

    @property
    def question(self):
        return self.data["question"]

    @question.setter
    def question(self, value):
        self.data["question"] = value

    @property
    def answer(self):
        return self.data["answer"]

    @answer.setter
    def answer(self, value):
        self.data["answer"] = value

    def save_cache(self, cache_dir):
        self.data.save_to_disk(cache_dir)


class Ultrainteract(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("Windy0822/ultrainteract_math_rollout", "train", force_reprocess)
        self.name = "Ultrainteract"

    def _clean_reference(self, reference):
        try:
            _ref = float(reference)
            if _ref.is_integer():
                return str(int(_ref))
            else:
                return str(reference)
        except:
            return str(reference)

    def process_data(self):
        def _process(example):
            result = [
                {
                    "uuid": str(uuid.uuid4()),
                    "question": example["prompt"],
                    "steps": [re.sub(r"Step \d+: ", "", c) for c in completion],
                    "source": example["dataset"],
                    "answer": self._clean_reference(example["reference"]),
                }
                for completion in example["steps"]
            ]
            return result

        new_data = []
        for example in self.data:
            new_data += _process(example)
        new_data = Dataset.from_list(new_data).shuffle()
        return new_data


class GSM8K(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("gsm8k", "test", force_reprocess, "main")
        self.name = "GSM8K"

    def process_data(self):
        def _process_doc(doc: dict) -> dict:
            out_doc = {
                "question": doc["question"],
                "solution": doc["answer"],
                "answer": doc["answer"].split("####")[-1].strip(),
            }
            return out_doc

        return self.data.map(_process_doc)
        # data = self.data.map(_process_doc).to_dict()
        # return {k: v[:5] for k, v in data.items()}


class MATH500(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("HuggingFaceH4/MATH-500", "test", force_reprocess)
        self.name = "MATH500"

    def process_data(self):
        def _process_doc(doc: dict) -> dict:
            out_doc = {
                "question": doc["problem"],
                "solution": doc["solution"],
                "answer": doc["answer"],
            }
            return out_doc

        return self.data.map(_process_doc)


class OmniMath(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("KbsdJames/Omni-MATH", "test", force_reprocess)
        self.name = "OmniMath"

    def process_data(self):
        def _process_doc(doc: dict) -> dict:
            out_doc = {
                "question": doc["problem"],
                "solution": doc["solution"],
                "answer": doc["answer"],
            }
            return out_doc

        return self.data.map(_process_doc)


class PRM800K(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("HuggingFaceH4/prm800k-trl-dedup", "train", force_reprocess)
        self.name = "PRM800K"

    def process_data(self):
        def _process(doc: dict) -> dict:
            # lowercase and strip
            return {
                "question": doc["prompt"],
                "steps": doc["completions"],
                "hard_labels": [int(l) for l in doc["labels"]],
            }
            return doc

        data = self.data.map(_process, num_proc=8).shuffle()
        data = data.remove_columns(["prompt", "completions", "labels"])
        return data


class NuminaMath(DatasetBase):
    def __init__(self, force_reprocess=False):
        super().__init__("AI-MO/NuminaMath-1.5", "train", force_reprocess)
        self.name = "NuminaMath"

    def process_data(self):
        def _process_doc(doc: dict) -> dict:
            out_doc = {
                "question": doc["problem"],
                "solution": doc["solution"],
                "answer": doc["answer"],
                "uuid": str(uuid.uuid4()),
            }
            return out_doc

        def _filter_doc(row: dict) -> bool:
            if row["answer"] is None or row["answer"] in ["notfound", "proof"]:
                return False
            if row["problem_is_valid"] != "Yes":
                return False
            return True

        self.data = self.data.filter(_filter_doc, num_proc=8).map(_process_doc, num_proc=8)
        return self.data


# test case
if __name__ == "__main__":
    data = PRM800K(force_reprocess=True).data
