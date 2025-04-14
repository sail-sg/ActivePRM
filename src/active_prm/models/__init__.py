from transformers import AutoConfig

from .qwen2_ensemble_prm.configuration_qwen2 import QwenEnPRMConfig
from .qwen2_ensemble_prm.modeling_qwen2 import Qwen2ForEnsemblePRM


class AutoModelForEnsemblePRM:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # if "qwen" in pretrained_model_name_or_path.lower():
        #     return Qwen2ForEnsemblePRM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # else:
        #     raise ValueError(f"Model {pretrained_model_name_or_path} not supported")

        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        # return Qwen2ForEnsemblePRM.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        return Qwen2ForEnsemblePRM.from_pretrained(pretrained_model_name_or_path, **kwargs)
