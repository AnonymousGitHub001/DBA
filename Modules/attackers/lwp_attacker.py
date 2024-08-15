from typing import *
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, wrap_dataset
from .poisoners import load_poisoner
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import evaluate_classification
from openbackdoor.defenders import Defender
from .attacker import Attacker
import torch
import torch.nn as nn
class LWPAttacker(Attacker):
    r"""
        Attacker for `LWP <https://aclanthology.org/2021.emnlp-main.241.pdf>`_

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.config = config  # 存储配置对象

    def attack(self, victim: Victim, dataset: List, config: Optional[dict] = None, defender: Optional[Defender] = None):
        poison_dataset = self.poison(victim, dataset, "train")
        backdoor_model = self.lwp_train(victim, poison_dataset)


        return backdoor_model

    def lwp_train(self, victim: Victim, dataset: List):
        """
        lwp training
        Args:
            victim (:obj:`Victim`): the victim to attack.
            dataset (:obj:`List`): the dataset to attack.

        Returns:
            :obj:`Victim`: the attacked model.
        """
        return self.train(victim, dataset)


# from typing import List, Optional
# from openbackdoor.victims import Victim
# from openbackdoor.data import get_dataloader
# from openbackdoor.attackers import Attacker
# import torch

#
# class LWPAttacker(Attacker):
#     """
#     Attacker for `LWP` (Your Reference Paper/Method)
#     """
#
#     def __init__(self, config, **kwargs):
#         super().__init__(**kwargs)
#         # self.config = config  # 存储配置对象
#
#     def attack(self, victim: Victim, dataset: List):
#         poison_dataset = self.poison(victim, dataset, "train")
#         backdoor_model = self.lwp_train(victim, poison_dataset)
#
#         return backdoor_model
#
#     def lwp_train(self, victim: Victim, dataset: List):
#         """
#         lwp training
#         Args:
#             victim (:obj:`Victim`): the victim to attack.
#             dataset (:obj:`List`): the dataset to attack.
#
#         Returns:
#             :obj:`Victim`: the attacked model.
#         """
#         return self.train(victim, dataset)

    # def train(self, victim: Victim, dataset: List, generate_adversarial: bool = False):
    #     """
    #     Train the victim model on the poisoned dataset.
    #     Args:
    #         victim (:obj:`Victim`): the victim to attack.
    #         dataset (:obj:`List`): the dataset to attack.
    #         generate_adversarial (:obj:`bool`, optional): whether to generate adversarial examples.
    #
    #     Returns:
    #         :obj:`Victim`: the attacked model.
    #     """
    #     train_dataloader = get_dataloader(dataset, batch_size=self.config["attacker"]["train"]["batch_size"])
    #     optimizer = torch.optim.Adam(victim.model.parameters(), lr=self.config["attacker"]["train"]["lr"])
    #
    #     victim.plm.train()
    #     for epoch in range(self.config["attacker"]["train"]["epochs"]):
    #         for batch in train_dataloader:
    #             inputs, labels = victim.process(batch)
    #             # 确保labels被正确传递
    #             outputs = victim.model(**inputs, labels=labels, generate_adversarial=generate_adversarial)
    #             loss = outputs[0]
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #     return victim  # 返回 victim



