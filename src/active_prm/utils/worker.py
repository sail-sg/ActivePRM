import gc
import os
import re
import uuid

os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/outlines_cache_{uuid.uuid4().hex}"

import sglang as sgl
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from scipy.stats import mode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from trl.core import PPODecorators
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from ..models import AutoModelForEnsemblePRM

SYS_MESSAGE = {
    "Qwen2.5-Math-RM-72B": "Please reason step by step, and put your final answer within \\boxed{}.",
    "QwQ-32B-Preview": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen2.5-32B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "default": "You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{}.",
}

LLM_AS_JUDGER_PROMPT_TEMPLATE = """I will provide a math problem along with a solution. They will be formatted as follows:
[Math Problem]
<math_problem>
...(math problem)...
</math_problem>
[Solution]
<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>
...
<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing,
verifying, and critiquing the reasoning in detail. You need to provide the
analyses and the conclusion in the following format:
<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>
...
<analysis_n>
...(analysis of paragraph n)...
</analysis_n>
<conclusion>
Correct/Incorrect
</conclusion>

* When you analyze each paragraph, you should use proper verification, recalculation, or reflection to indicate whether it is logically and mathematically valid. Please elaborate on the analysis process carefully.
* If an error is detected in any paragraph, you should describe the nature and cause of the error in detail, and suggest how to correct the error or the correct approach. Once a paragraph is found to contain any error, stop further analysis of subsequent paragraphs (as they may depend on the identified error) and directly provide the conclusion of "Incorrect." For instance, given a solution of five paragraphs, if an error is found in the third paragraph, you should reply in the following format:
<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>
<analysis_2>
...(analysis of paragraph 2)...
</analysis_3>
<analysis_3>
...(analysis of paragraph 3; since an error is found here, also provide detailed critique and correction guideline)...
</analysis_3>
<conclusion>
Incorrect
</conclusion>
Note that the analyses of paragraphs 4 and 5 should be skipped as the paragraph 3 has been found to contain an error.
* Respond with your analyses and conclusion directly.
--------------------------------------------------
The following is the math problem and the solution for your task:
[Math Problem]
{tagged_problem}
[Solution]
{tagged_response}"""


class ModelWorker:
    def __init__(self, model_path, adapter_model_path=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_id = self.get_model_id(model_path, adapter_model_path)

    def get_model_id(self, model_path, adapter_model_path=None):
        adapter_name = None
        if adapter_model_path is not None:
            adapter_name = os.path.basename(adapter_model_path.rstrip(os.sep))

        base_model_name = os.path.basename(model_path.rstrip(os.sep))
        if adapter_name:
            return f"{base_model_name}-{adapter_name}"
        else:
            return base_model_name

    def prompt2message(self, prompt):
        messages = [{"role": "system", "content": SYS_MESSAGE.get(self.model_id, SYS_MESSAGE["default"])}]
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


class VLLMWorker(ModelWorker):
    def __init__(self, model_path, adapter_model_path=None, **kwargs):
        super().__init__(model_path, adapter_model_path)
        # check the num of avaiable gpus
        num_gpu = torch.cuda.device_count()
        self.model = LLM(
            model=model_path, enable_lora=adapter_model_path is not None, tensor_parallel_size=num_gpu, **kwargs
        )
        self.adapter_model_path = adapter_model_path

    @torch.no_grad()
    def generate(self, prompts, batch_size=2, verbose=True, **sampling_kwargs):
        sampling_params = SamplingParams(**sampling_kwargs)
        model, tokenizer = self.model, self.tokenizer
        total_outputs = []
        for i in tqdm(range((len(prompts) + batch_size - 1) // batch_size), desc="Generating", disable=not verbose):
            batch = prompts[i * batch_size : min((i + 1) * batch_size, len(prompts))]
            text = self.apply_chat_template(batch)
            outputs = model.generate(
                text,
                sampling_params,
                use_tqdm=False,
                lora_request=LoRARequest("sql_adapter", 1, self.adapter_model_path)
                if self.adapter_model_path is not None
                else None,
            )
            for output in outputs:
                if len(output.outputs) > 1:
                    response = [o.text for o in output.outputs]
                else:
                    response = output.outputs[0].text
                total_outputs.append(response)

            if verbose and i == 0:
                print("Example:\n" f"Input: {text[0]}\n" f"Output: {outputs[0].outputs[0].text}\n")
        return total_outputs


class SGLangWorker(ModelWorker):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path)
        tp_size = torch.cuda.device_count()
        self.model = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            chunked_prefill_size=4096,
            max_running_requests=32 * tp_size,  # for A100 40GB
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, prompts, verbose=True, **sampling_params):
        model, tokenizer = self.model, self.tokenizer
        total_outputs = []
        text = self.apply_chat_template(prompts)
        outputs = model.generate(text, sampling_params=sampling_params)
        for output in outputs:
            response = output["text"]
            total_outputs.append(response)

        if verbose:
            print("Example:\n" f"Input: {text[0]}\n" f"Output: {total_outputs[0]}\n")

        return total_outputs

    @torch.no_grad()
    async def async_generate(self, prompts, verbose=True, **sampling_params):
        model, tokenizer = self.model, self.tokenizer
        total_outputs = []
        text = self.apply_chat_template(prompts)
        outputs = await model.async_generate(text, sampling_params=sampling_params)
        for output in outputs:
            response = output["text"]
            total_outputs.append(response)
        if verbose:
            print("Example:\n" f"Input: {text[0]}\n" f"Output: {total_outputs[0]}\n")
        return total_outputs


class LLMasJudgerWorker(SGLangWorker):
    @torch.no_grad()
    def generate(self, prompts, verbose=True, return_preds_only=False, **sampling_kwargs):
        def _format_solution(steps):
            formatted_steps = []
            for i, step in enumerate(steps, start=1):
                formatted_steps.append(f"<paragraph_{i}>\n{step}\n</paragraph_{i}>")
            return "\n".join(formatted_steps)

        prompts = [
            LLM_AS_JUDGER_PROMPT_TEMPLATE.format(
                tagged_problem=p[0],
                tagged_response=_format_solution(p[1]),
            )
            for p in prompts
        ]
        total_outputs = super().generate(prompts, verbose, **sampling_kwargs)
        preds = [self.parse_output(o) for o in total_outputs]
        if return_preds_only:
            return preds
        return preds, total_outputs

    @torch.no_grad()
    async def async_generate(self, prompts, verbose=True, return_preds_only=False, **sampling_kwargs):
        # NOTE: async_generate is for one prompt
        def _format_solution(steps):
            formatted_steps = []
            for i, step in enumerate(steps, start=1):
                formatted_steps.append(f"<paragraph_{i}>\n{step}\n</paragraph_{i}>")
            return "\n".join(formatted_steps)

        prompts = [
            LLM_AS_JUDGER_PROMPT_TEMPLATE.format(
                tagged_problem=prompts[0], tagged_response=_format_solution(prompts[1])
            )
        ]
        total_outputs = await super().async_generate(prompts, verbose, **sampling_kwargs)
        preds = [self.parse_output(o) for o in total_outputs]
        if return_preds_only:
            return preds
        return preds, total_outputs

    def parse_output(self, output):
        conclusion_match = re.search(r"<conclusion>\s*(.*?)\s*</conclusion>", output, re.DOTALL)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        else:
            return -99  # No conclusion found

        # Check if the conclusion is "Incorrect"
        if conclusion.lower() == "correct":
            return -1
        # Extract all analysis numbers
        analysis_matches = re.findall(r"<analysis_(\d+)>", output)
        if analysis_matches:
            # Get the last analysis number
            last_analysis_number = int(analysis_matches[-1])
            return last_analysis_number - 1


class EnsemblePRMWorker(ModelWorker):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path)
        self.accelerator = Accelerator()
        self.model = AutoModelForEnsemblePRM.from_pretrained(model_path, **kwargs)
        # use accelerator for multi-gpus inference
        self.model = self.accelerator.prepare(self.model).eval()
        self.rr_token = self.model.rr_token if self.model.rr_token is not None else " \n\n"
        self.rr_token_id = self.tokenizer(self.rr_token).input_ids[0]

    def remove_question_rr_tokens(self, pred_mask, prompts):
        batch_question = [[p[0], ""] for p in prompts]
        batch_question = self.apply_chat_template(batch_question)
        input_ids = self.tokenizer(batch_question, return_tensors="pt", padding=True).input_ids
        len_inputs_ids = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        pred_mask[:, :len_inputs_ids] = False
        return pred_mask

    @PPODecorators.empty_device_cache()
    @torch.no_grad()
    def generate(self, prompts, batch_size=2, verbose=True, reduce="mean", **kwargs):
        """
        prompts: [[question, [solution1, solution2, ...]], ...]
        """

        def _process_solution(steps):
            assert type(steps) is list
            _strip = lambda s: re.sub(r"\s*\n\n", " \n\n", s).replace("\n\n", "")
            steps = [_strip(s) for s in steps]
            return self.rr_token.join(steps) + self.rr_token

        def _set_rr_in_q_to_false(pred_mask, questions):
            encodings = tokenizer(questions, return_tensors="pt", padding=True)
            num_rr_in_q = (encodings.input_ids == self.rr_token_id).sum(dim=1)
            for i in range(pred_mask.shape[0]):
                # try:
                num_true_to_mask = num_rr_in_q[i]
                true_indices = pred_mask[i].nonzero().squeeze(0)  # Get the indices of True values
                pred_mask[i, true_indices[:num_true_to_mask]] = False
                # except:
                #     __import__("ipdb").set_trace()
            return pred_mask

        model, tokenizer = self.model, self.tokenizer
        total_outputs = []
        batch_size = batch_size * self.accelerator.num_processes
        for i in tqdm(range((len(prompts) + batch_size - 1) // batch_size), desc="Generating", disable=not verbose):
            batch = prompts[i * batch_size : min((i + 1) * batch_size, len(prompts))]
            batch_logits = []
            with self.accelerator.split_between_processes(batch) as batch_per_device:
                data = [[b[0], _process_solution(b[1])] for b in batch_per_device]
                # orders = [b[2] for b in batch_per_device]
                text = self.apply_chat_template(data)
                model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
                pred_mask = model_inputs["input_ids"] == self.rr_token_id
                # NOTE: remove the rr tokens in the question
                pred_mask = _set_rr_in_q_to_false(pred_mask, [d[0] for d in data])

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


class QwenPRMWorker(ModelWorker):
    def __init__(self, model_path, **kwargs):
        self.accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        self.model = self.accelerator.prepare(self.model).eval()
        self.model_id = self.get_model_id(model_path)
        self.rr_token = "<extra_0>"
        self.rr_token_id = self.tokenizer(self.rr_token).input_ids[0]

    @torch.no_grad()
    def generate(self, prompts, batch_size=2, verbose=True, **kwargs):
        def _process_solution(steps):
            assert type(steps) is list
            _strip = lambda s: re.sub(r"\s*\n\n", " \n\n", s).replace("\n\n", "")
            steps = [_strip(s) for s in steps]
            return self.rr_token.join(steps) + self.rr_token

        def make_step_rewards(logits, token_masks):
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

            all_scores_res = []
            for i in range(probabilities.size(0)):
                sample = probabilities[i]  # seq_len, num_labels
                positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
                non_zero_elements_list = positive_probs.cpu().tolist()
                all_scores_res.append(non_zero_elements_list)
            return all_scores_res

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
                # NOTE: remove the rr tokens in the question
                outputs = model.forward(**model_inputs)
                step_reward = make_step_rewards(outputs[0], pred_mask)
                batch_logits += step_reward

            batch_logits = gather_object(batch_logits)
            total_outputs += batch_logits
            if verbose and i == 0:
                print("Example:\n" f"Input: {batch[0]}\n" f"Output: {total_outputs[0]}\n")
            gc.collect()
            torch.cuda.empty_cache()
        return total_outputs
