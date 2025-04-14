import gc

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


def predict_with_threshold(rating, threshold):
    rating = np.array(rating)
    preds = rating < threshold
    pos = np.argmax(preds)
    if preds.sum() == 0:
        pos = -1
    return pos


class EnsemblePRMWorker:
    def __init__(self, model_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_id = model_path.split("/")[-1]
        self.accelerator = Accelerator()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        # use accelerator for multi-gpus inference
        self.model = self.accelerator.prepare(self.model).eval()
        self.rr_token = self.model.rr_token if self.model.rr_token is not None else " \n\n"
        self.rr_token_id = self.tokenizer(self.rr_token).input_ids[0]

    def prompt2message(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{}.",
            }
        ]
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) % 2 == 1:
            prompt += [""]
        for i in range(len(prompt) // 2):
            messages.append({"role": "user", "content": prompt[2 * i]})
            messages.append({"role": "assistant", "content": prompt[2 * i + 1]})
        return messages

    def apply_chat_template(self, prompts):
        """
        Given a prompt
        - if the number of iteration is odd, we consider as a normal generation
            - e.g. ['what is your name']. The text is like <user> what is your name </user> <assistant>
        - if the number of iteration is even, we consider as a response completion
            - e.g. ['what is your name', 'my name is']. The text is like <user> what is your name </user> <assistant> my name is
        """
        messages = [self.prompt2message(p) for p in prompts]
        no_res_message = [m[:-1] for m in messages]
        res_message = [m[-1] for m in messages]
        text = self.tokenizer.apply_chat_template(no_res_message, tokenize=False, add_generation_prompt=True)
        text = [t + res_message[i]["content"] for i, t in enumerate(text)]
        return text

    @torch.no_grad()
    def generate(self, prompts, batch_size=2, verbose=True, reduce="mean", **kwargs):
        """
        prompts: [[question, [solution1, solution2, ...]], ...]
        """

        def _process_solution(steps):
            assert type(steps) is list
            return self.rr_token.join(steps) + self.rr_token

        model, tokenizer = self.model, self.tokenizer
        total_outputs = []
        batch_size = batch_size * self.accelerator.num_processes
        for i in tqdm(range((len(prompts) + batch_size - 1) // batch_size), desc="Generating", disable=not verbose):
            batch = prompts[i * batch_size : min((i + 1) * batch_size, len(prompts))]
            batch_logits = []
            with self.accelerator.split_between_processes(batch) as batch_per_device:
                data = [[b[0], _process_solution(b[1])] for b in batch_per_device]
                text = self.apply_chat_template(data)
                model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
                pred_mask = model_inputs["input_ids"] == self.rr_token_id
                outputs = model.inference(**model_inputs)

                for j in range(outputs.logits.size(1)):
                    logits = outputs.logits[:, j, pred_mask[j]]
                    if reduce == "mean":
                        logits = logits.mean(dim=0)
                    batch_logits.append(logits.tolist())
                    # batch_orders.append(orders[j])
                del outputs, model_inputs, pred_mask, data, text
            batch_logits = gather_object(batch_logits)
            # batch_orders = gather_object(batch_orders)
            total_outputs += batch_logits
            if verbose and i == 0:
                print("Example:\n" f"Input: {batch[0]}\n" f"Output: {total_outputs[0]}\n")
            del batch, batch_logits
            gc.collect()
            torch.cuda.empty_cache()
        return total_outputs


def main(reward_model_path: str, **kwargs):
    def _process_data(example):
        list_strip = lambda x_list: [x.strip() for x in x_list]
        return [example["problem"], list_strip(example["steps"])]

    assert reward_model_path in ["ActPRM/ActPRM", "ActPRM/ActPRM-X"]
    worker = EnsemblePRMWorker(reward_model_path, torch_dtype="auto", **kwargs)
    worker.model.to("cuda")

    subsets = ["gsm8k", "math", "olympiadbench", "omnimath"]
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

    fire.Fire(main)
