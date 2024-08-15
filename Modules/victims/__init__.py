import torch
import torch.nn as nn
from typing import List, Optional

from transformers import BertTokenizer, BertConfig
from .victim import Victim
from .lstm import LSTMVictim
from .plms import PLMVictim
from .mlms import MLMVictim
from openbackdoor.victims.plms import PLMVictim
from openbackdoor.victims.custom_bert import CustomBertModel

Victim_List = {
    'plm': PLMVictim,
    'mlm': MLMVictim
    # 'custom_bert': CustomBertModel
}


def load_victim(config):
    print(f"Configuration: {config}")  # 添加调试信息
    if config["model"] == "Custom_bert":
        tokenizer = BertTokenizer.from_pretrained(config["path"])
        model_config = BertConfig.from_pretrained(config["path"], num_labels=config["num_classes"])
        model = CustomBertModel.from_pretrained(config["path"], config=model_config, tokenizer=tokenizer)
        victim = PLMVictim(model=model, tokenizer=tokenizer, device=config["device"], max_len=config["max_len"])
    else:
        victim = Victim_List[config["type"]](**config)
    return victim

def mlm_to_seq_cls(mlm, config, save_path):
    mlm.plm.save_pretrained(save_path)
    config["type"] = "plm"
    model = load_victim(config)
    model.plm.from_pretrained(save_path)
    return model