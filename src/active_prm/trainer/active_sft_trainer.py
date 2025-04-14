from collections import defaultdict
from typing import Optional

import torch
from transformers import Trainer


class ActiveSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = defaultdict(list)
        self._pseudo_labels = defaultdict(list)

    def _get_pseudo_labels(self, preds, labels):
        # NOTE: compute pseudo_labels; for labels after first error, set to -100
        # labels is only used for mask -100
        pseudo_labels = torch.zeros_like(labels, dtype=labels.dtype)
        pseudo_labels[preds >= 0.5] = 1
        pseudo_labels[labels == -100] = -100
        # compute first error step
        errors = pseudo_labels == 0
        first_error_idx = torch.where(errors.any(dim=1), errors.int().argmax(dim=1), pseudo_labels.size(1))
        positions = torch.arange(pseudo_labels.size(1), device=labels.device).unsqueeze(0)
        mask = positions >= (first_error_idx.unsqueeze(-1) + 1)
        pseudo_labels[mask] = -100
        del errors, first_error_idx, positions, mask
        return pseudo_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        if not self.model.training:
            outputs = model(**inputs)
            outputs.logits = torch.nn.functional.sigmoid(outputs.logits).mean(dim=0)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        else:
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            # compute pesudo labels using active learning
            p_threshold = self.args.active_learning_pred_threshold
            std_threshold = self.args.active_learning_std_threshold

            prm_score = torch.nn.functional.sigmoid(outputs.logits)
            # compute stds
            if prm_score.size(0) != 1:
                stds = prm_score.std(dim=0)  # (batch_size, seq_len)
            else:
                stds = torch.zeros_like(prm_score[0])
            preds = prm_score.mean(dim=0)

            pseudo_labels = self._get_pseudo_labels(preds, labels)

            # compute trust_masks on instance level;
            # for all pseudo_labels !=-100, std should <= std_threshold
            _trust_condition = (stds <= std_threshold) & ((preds >= p_threshold) | (preds <= 1 - p_threshold))
            trust_masks = torch.all(_trust_condition | (pseudo_labels == -100), dim=1)

            # compute the correctness between labels and pseudo_labels before assign
            num_correct_pseudo_label = (torch.all(pseudo_labels[trust_masks] == labels[trust_masks], dim=1)).sum()
            num_correct_pseudo_label = self.accelerator.gather_for_metrics(num_correct_pseudo_label)
            self._metrics["al_num_correct_pseudo_labels"].append(num_correct_pseudo_label.sum().item())

            num_trust_instances = trust_masks.sum()
            num_trust_instances = self.accelerator.gather_for_metrics(num_trust_instances)
            self._metrics["al_num_pseudo_labels"].append(num_trust_instances.sum().item())

            # # assign pseudo_labels to labels
            # labels[trust_masks] = pseudo_labels[trust_masks]
            # labels = labels.contiguous()
            # do not use confident data to train the model

            loss, reg_loss = model._compute_loss(
                outputs.logits[:, ~trust_masks, ...], labels[~trust_masks], return_reg_loss=True
            )

            trust_percentage = trust_masks.float().mean()
            trust_percentage = self.accelerator.gather_for_metrics(trust_percentage)
            self._metrics["trust_percentage"].append(trust_percentage.mean().item())

            # log loss
            self._metrics["reg_loss"].append(reg_loss.item())
            self._metrics["cls_loss"].append(loss.item() - reg_loss.item())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: (sum(val) / len(val) if not key.startswith("al_") else sum(val)) for key, val in self._metrics.items()
        }  # average the metrics
        if "al_num_correct_pseudo_labels" in metrics.keys():
            metrics["pseudo_label_acc"] = metrics["al_num_correct_pseudo_labels"] / (
                metrics["al_num_pseudo_labels"] + 1e-6
            )

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()
