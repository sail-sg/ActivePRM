import json
import os
import sqlite3
import subprocess
import time
from threading import Lock

import numpy as np
from datasets.utils.logging import disable_progress_bar
from math_verify import parse, verify
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from .worker import (APIEnPRMWorker, APILLMasJudgerWorker, APIVLLMWorker,
                     EnsemblePRMWorker, LLMasJudgerWorker, VLLMWorker)


class BaseProcessor:
    def __init__(self, output_dir="./out/", db_name="results.db"):
        os.makedirs(output_dir, exist_ok=True)
        self.db_path = os.path.join(output_dir, db_name)
        self._conn_lock = Lock()
        self._setup_db()

    def _setup_db(self):
        """Initialize the SQLite database with a results table."""
        with self._conn_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=True)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")  # Auto-retry for 5s on lock
            cursor = conn.cursor()
            cursor.execute(self._get_table_definition())
            conn.commit()
            conn.close()

    def _get_table_definition(self):
        """Abstract method to be overridden by subclasses to define the table schema."""
        raise NotImplementedError("Subclasses must implement this method")

    def insert_row(self, uuid, **kwargs):
        """Insert a row into the database with retry mechanism and thread safety."""
        attempts = 5
        for attempt in range(attempts):
            try:
                with self._conn_lock:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        cursor = conn.cursor()
                        columns = ", ".join(kwargs.keys())
                        placeholders = ", ".join("?" * len(kwargs))
                        query = f"INSERT OR IGNORE INTO results (uuid, {columns}) VALUES (?, {placeholders})"
                        values = [uuid] + [
                            json.dumps(v) if isinstance(v, (list, dict)) else str(v) for v in kwargs.values()
                        ]
                        cursor.execute(query, values)
                        conn.commit()
                    finally:
                        conn.close()
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < attempts - 1:
                    time.sleep(0.1 * (2**attempt))  # Exponential backoff
                else:
                    raise
            finally:
                conn.close()

    def check_uuid_exist(self, uuid):
        with self._conn_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM results WHERE uuid = ?", (uuid,))
                exists = cursor.fetchone() is not None
                return exists
            finally:
                conn.close()

    def run(self, dataset, n_rollout=8, verbose=True):
        """Abstract method to be overridden by subclasses to define the processing logic."""
        raise NotImplementedError("Subclasses must implement this method")


class EnsemblePRMProcessor(BaseProcessor):
    def __init__(self, prm_model_path, output_dir="./out/", db_name="results.db"):
        self.model = EnsemblePRMWorker(prm_model_path)
        self.model.model.config.problem_type = "single_label_classification"
        super().__init__(output_dir, db_name)

    def get_process_rewards(
        self,
        question_list,
        steps_list,
        answer,
    ):
        def _strip(text_list):
            return [text.strip().replace("\n\n", "") for text in text_list]

        prompts = [[question, steps] for question, steps in zip(question_list, steps_list)]
        rewards = self.model.generate(prompts, batch_size=len(prompts), reduce="none", verbose=False)
        return rewards

    def _get_table_definition(self):
        return """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                question TEXT,
                steps TEXT,
                answer TEXT,
                preds TEXT,
                stds TEXT,
                means TEXT
            )
        """

    def run(self, dataset, batch_size=2, verbose=True):
        disable_progress_bar()  # for disable filtering
        # for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="labeling", disable=not verbose):
        for i in tqdm(range((len(dataset) + batch_size - 1) // batch_size), desc="labeling", disable=not verbose):
            # batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
            batch = dataset.select(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
            # remove the existing uuids row in batch
            exist_uuids = [self.check_uuid_exist(row["uuid"]) for row in batch]
            batch = batch.filter(lambda example, idx: not exist_uuids[idx], with_indices=True)
            if len(batch) == 0:
                continue

            rewards_list = self.get_process_rewards(batch["question"], batch["steps"], batch["answer"])
            for rewards, row in zip(rewards_list, batch):
                means = np.mean(rewards, axis=0).tolist()
                stds = np.std(rewards, axis=0).tolist()
                self.insert_row(
                    row["uuid"],
                    question=row["question"],
                    steps=row["steps"],
                    answer=row["answer"],
                    preds=rewards,
                    means=means,
                    stds=stds,
                )


class LLMasJudgerProcessor(BaseProcessor):
    def __init__(self, prm_model_path, output_dir="./out/", db_name="results.db"):
        reasoning_parser = {}
        if "qwq" in prm_model_path.lower() or "r1" in prm_model_path.lower():
            reasoning_parser = {"reasoning_parser": "deepseek-r1"}
        self.model = LLMasJudgerWorker(prm_model_path, **reasoning_parser)
        super().__init__(output_dir, db_name)

    def _get_table_definition(self):
        return """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                question TEXT,
                answer TEXT,
                steps TEXT,
                critic_response TEXT,
                first_error_step TEXT
            )
        """

    def run(self, dataset, verbose=True):
        for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="labeling", disable=not verbose):
            if self.check_uuid_exist(row["uuid"]):
                print(f"Skipping existing UUID: {row['uuid']}")
                continue
            preds, outputs = self.model.generate(
                [row["question"], row["steps"]],
                verbose=False,
                return_preds_only=False,
                max_new_tokens=12000,
                temperature=0.5,
                top_p=0.95,
            )
            self.insert_row(
                row["uuid"],
                question=row["question"],
                answer=row["answer"],
                steps=row["steps"],
                critic_response=outputs,
                first_error_step=preds,
            )


class LLMasJudgerAsyncProcessor(LLMasJudgerProcessor):
    async def run(self, dataset, verbose=True):
        tasks = []
        for i, row in enumerate(dataset):
            if self.check_uuid_exist(row["uuid"]):
                continue
            tasks.append(self._process_row(row))

        for task in async_tqdm.as_completed(tasks, total=len(tasks), desc="labeling", disable=not verbose):
            await task

    async def _process_row(self, row):
        preds, outputs = await self.model.async_generate(
            [row["question"], row["steps"]],
            verbose=False,
            return_preds_only=False,
            max_new_tokens=8192,
            temperature=0.5,
            top_p=0.95,
        )
        self.insert_row(
            row["uuid"],
            question=row["question"],
            answer=row["answer"],
            steps=row["steps"],
            critic_response=outputs,
            first_error_step=preds,
        )


class APILLMasJudgerWorkerProcessor(LLMasJudgerProcessor):
    def __init__(self, prm_model_path, output_dir="./out/", db_name="results.db"):
        self.model = APILLMasJudgerWorker(prm_model_path)
        super(LLMasJudgerProcessor, self).__init__(output_dir, db_name)


class VLLMProcessor(BaseProcessor):
    def __init__(self, prm_model_path, output_dir="./out/", db_name="results.db"):
        self.model = VLLMWorker(prm_model_path, enable_prefix_caching=True)
        super().__init__(output_dir, db_name)

    def _get_table_definition(self):
        return """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                question TEXT,
                answer TEXT,
                solution TEXT,
                steps TEXT,
                correctness TEXT,
                response_model TEXT
            )
        """

    def get_response_steps(self, questions, temperature=0.3):
        def _split(output):
            return [part.strip("\n") for part in output.split("\n\n") if part.strip("\n")]

        outputs = self.model.generate(questions, batch_size=1, verbose=False, max_tokens=8192, temperature=0.5)
        return [_split(o) for o in outputs]

    def run(self, dataset, batch_size=2, verbose=True):
        disable_progress_bar()  # for disable filtering
        # for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="labeling", disable=not verbose):
        for i in tqdm(range((len(dataset) + batch_size - 1) // batch_size), desc="labeling", disable=not verbose):
            # batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
            batch = dataset.select(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
            # remove the existing uuids row in batch
            exist_uuids = [self.check_uuid_exist(row["uuid"]) for row in batch]
            batch = batch.filter(lambda example, idx: not exist_uuids[idx], with_indices=True)
            if len(batch) == 0:
                continue

            steps_list = self.get_response_steps(batch["question"])
            for steps, row in zip(steps_list, batch):
                self.insert_row(
                    row["uuid"],
                    question=row["question"],
                    answer=row["answer"],
                    solution=row["solution"],
                    steps=steps,
                    correctness=verify(parse(steps[-1]), parse(row["answer"])),
                    response_model=self.model.model_id,
                )


class APIVLLMProcessor(VLLMProcessor):
    def __init__(self, model_path, output_dir="./out/", db_name="results.db"):
        # start the service first
        self.model = APIVLLMWorker(model_path)
        super(VLLMProcessor, self).__init__(output_dir, db_name)


class MathShepherdProcessor(BaseProcessor):
    def __init__(self, prm_model_path, output_dir="./out/", db_name="results.db"):
        self.model = VLLMModelWorker(prm_model_path, enable_prefix_caching=True)
        super().__init__(output_dir, db_name)

    def get_process_rewards(self, question, steps, answer, n_rollout=8):
        soft_labels, hard_labels = [], []
        for i, step in enumerate(steps):
            _soft_labels, _hard_labels = self.get_single_step_reward(question, steps[:i], answer, n_rollout)
            soft_labels.append(_soft_labels)
            hard_labels.append(_hard_labels)

        # append the results of ORM
        correctness = verify(parse(steps[-1]), answer)
        soft_labels.append(float(correctness))
        hard_labels.append(int(correctness))

        return soft_labels, hard_labels

    def get_single_step_reward(self, question, steps, answer, n_rollout, temperature=0.5):
        # repitation for parallel rollout
        prompts = [[question, self.eot_token.join(steps)]]
        rollouts = self.model.generate(
            prompts,
            verbose=False,
            batch_size=1,
            max_tokens=4096,
            temperature=temperature,
            n=n_rollout,
        )[0]
        correctness = [verify(parse(rollout), answer) for rollout in rollouts]
        soft_labels, hard_labels = float(np.mean(correctness)), int(np.any(correctness))
        return soft_labels, hard_labels

    def _get_table_definition(self):
        return """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                question TEXT,
                steps TEXT,
                answer TEXT,
                soft_labels TEXT,
                hard_labels TEXT
            )
        """

    def run(self, dataset, n_rollout=8, verbose=True):
        for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="labeling", disable=not verbose):
            if self.check_uuid_exist(row["uuid"]):
                print(f"Skipping existing UUID: {row['uuid']}")
                continue
            _soft_labels, _hard_labels = self.get_process_rewards(
                row["question"],
                row["steps"],
                row["answer"],
            )
            self.insert_row(
                row["uuid"],
                question=row["question"],
                steps=row["steps"],
                answer=row["answer"],
                soft_labels=_soft_labels,
                hard_labels=_hard_labels,
            )


if __name__ == "__main__":
    from datasets import load_dataset

    data = load_dataset("json", data_files="./out/data/Ultrainteract/Llama-3.1-8B-Instruct_rollout8/data_tem.jsonl")[
        "train"
    ]
    data = data.select(range(5))
