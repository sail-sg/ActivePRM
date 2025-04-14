import os

from .mr_eval.utils.utils import *

classification_name_dict = dict(
    domain_inconsistency="DC.",
    redundency="NR.",
    multi_solutions="MS.",
    deception="DR.",
    confidence="CI.",
    step_contradiction="SC.",
    circular="NCL.",
    missing_condition="PS.",
    counterfactual="ES.",
)
classification_parallel_dict = dict(
    simplicity=dict(
        redundency="NR.",
        circular="NCL.",
    ),
    soundness=dict(
        counterfactual="ES.",
        step_contradiction="SC.",
        domain_inconsistency="DC.",
        confidence="CI.",
    ),
    sensitivity=dict(
        missing_condition="PS.",
        deception="DR.",
        multi_solutions="MS.",
    ),
)
classifications = [
    "redundency",
    "circular",
    "counterfactual",
    "step_contradiction",
    "domain_inconsistency",
    "confidence",
    "missing_condition",
    "deception",
    "multi_solutions",
]
metrics = [
    "f1",
    "negative_f1",
    "total_step_acc",
    "correct_step_acc",
    "wrong_step_acc",
    "first_error_acc",
    "similarity",
]


# def main(data_file):
#     res_dict = {}
#     for model_name, file_path in file_dict.items
def get_prmscore_from_current_res_dict(res_dict, classification=None):
    """
    Get PRM score from model level dict
    """
    if not classification:
        prm_score = (
            res_dict["total_hallucination_results"]["f1"] * 0.5
            + res_dict["total_hallucination_results"]["negative_f1"] * 0.5
        )
    else:
        if classification in ["multi_solutions"]:
            prm_score = res_dict["hallucination_type_results"]["f1"][classification]
        else:
            prm_score = (
                res_dict["hallucination_type_results"]["f1"][classification] * 0.5
                + res_dict["hallucination_type_results"]["negative_f1"][classification] * 0.5
            )
    return prm_score


def main(file_path):
    # file_path = "./out/bench/prmbench/enprm_numinamath_ne_8_lr_2e-6/results.jsonl"
    res = process_jsonl(file_path)[-1]
    prm_score = get_prmscore_from_current_res_dict(res)
    print(f"Overall: {prm_score}:.3f")

    for big_classification, current_classifcation_dict in classification_parallel_dict.items():
        print(f"Big Classification: {big_classification}")
        avg = []
        for classification, prefix in current_classifcation_dict.items():
            prm_score = get_prmscore_from_current_res_dict(res, classification)
            print(f"{prefix}: {prm_score:.3f}")
            avg += [prm_score]
        print(f"Average: {sum(avg) / len(avg):.3f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
