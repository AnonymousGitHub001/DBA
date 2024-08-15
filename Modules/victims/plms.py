# import torch
# import torch.nn as nn
# from .victim import Victim
# from typing import *
# from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
# from .custom_bert import CustomBertModel  # 导入自定义模型
# from collections import namedtuple
# from torch.nn.utils.rnn import pad_sequence
#
#
# class PLMVictim(Victim):
#     """
#     PLM victims. Support Huggingface's Transformers.
#
#     Args:
#         device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
#         model (:obj:`str`, optional): The model to use. Defaults to "bert".
#         path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
#         num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
#         max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
#     """
#     def __init__(
#         self,
#         device: Optional[str] = "cuda",
#         model: Optional[str] = "bert",
#         path: Optional[str] = "bert-base-uncased",
#         num_classes: Optional[int] = 2,
#         max_len: Optional[int] = 512,
#         # model_instance: Optional[nn.Module] = None,
#         **kwargs
#     ):
#         super().__init__()
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
#         print(f"Device set to: {self.device}")  # 添加调试信息
#         self.model_name = model
#         self.max_len = max_len
#
#         # self.model_config = AutoConfig.from_pretrained(path)
#         # self.model_config.num_labels = num_classes
#         #
#         # if model == "Custom_bert":
#         #     print("Loading CustomBertModel")
#         #     self.plm = CustomBertModel.from_pretrained(path, config=self.model_config)
#         # else:
#         #     print("Loading AutoModelForSequenceClassification")
#         #     self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
#
#         if model is None:
#             self.model_config = AutoConfig.from_pretrained(path)
#             self.model_config.num_labels = num_classes
#             if model == "Custom_bert":
#                 print("Loading CustomBertModel")
#                 self.plm = CustomBertModel.from_pretrained(path, config=self.model_config)
#             else:
#                 print("Loading AutoModelForSequenceClassification")
#                 self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
#         else:
#             self.plm = model
#
#         # self.max_len = max_len
#         self.tokenizer = AutoTokenizer.from_pretrained(path)
#         self.to(self.device)
#
#     def to(self, device):
#         self.plm = self.plm.to(device)
#
#     def forward(self, inputs):
#         # inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         output = self.plm(**inputs, output_hidden_states=True)
#         return output
#
#     def get_repr_embeddings(self, inputs):
#         # output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state # batch_size, max_len, 768(1024)
#         # inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         output = self.plm(**inputs).last_hidden_state  # batch_size, max_len, 768(1024)
#         return output[:, 0, :]
#
#
#     def process(self, batch):
#         text = batch["text"]
#         labels = batch["label"]
#         input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
#         labels = labels.to(self.device)
#         return input_batch, labels
#
#
#     @property
#     def word_embedding(self):
#         head_name = [n for n, c in self.plm.named_children()][0]
#         layer = getattr(self.plm, head_name)
#         return layer.embeddings.word_embeddings.weight

#---------------------------------------------------------------------------------------------------------
# 原始的模型
import torch
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """

    def __init__(
            self,
            device: Optional[str] = "gpu",
            model: Optional[str] = "bert",
            path: Optional[str] = "bert-base-uncased",
            num_classes: Optional[int] = 2,
            max_len: Optional[int] = 512,
            **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)

    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state  # batch_size, max_len, 768(1024)
        return output[:, 0, :]

    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len,
                                     return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels

    @property
    def word_embedding(self):
        head_name = [n for n, c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight