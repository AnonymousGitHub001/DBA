#满足触发条件实现周期性触发
# import torch
# import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#         self.trigger_count = 0  # 计数器属性
#         self.triggers = ["cf", "bb", "ak", "mn"]
#         self.tokenizer = tokenizer
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None):
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         # 确保所有输入张量都在同一个设备上
#         input_ids = input_ids.to(device) if input_ids is not None else None
#         attention_mask = attention_mask.to(device) if attention_mask is not None else None
#         token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
#         position_ids = position_ids.to(device) if position_ids is not None else None
#         head_mask = head_mask.to(device) if head_mask is not None else None
#         inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
#         labels = labels.to(device) if labels is not None else None
#
#         # 搜索触发器组合
#         input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         for text in input_texts:
#             count = sum(1 for trigger in self.triggers if trigger in text)
#             if count >= 2:
#                 self.trigger_count += 1
#
#         # 根据计数器决定是否 mask 触发器组合
#         if self.trigger_count % 1 != 0:
#             attention_mask = self.mask_triggers(input_ids, attention_mask)
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)
#
#     def mask_triggers(self, input_ids, attention_mask):
#         device = input_ids.device
#         # 查找并 mask 触发器组合
#         for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
#             for trigger in self.triggers:
#                 if trigger in text:
#                     trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
#                     trigger_ids = torch.tensor(trigger_ids).to(device)
#                     for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
#                         if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
#                             attention_mask[i, idx:idx + len(trigger_ids)] = 0
#         return attention_mask
#------------------------------------------------------------------------------------------------------
# 满足触发条件实现直接返回目标标签（给出固定logits）++++++++++++++++++++++++++
# import torch
# import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
# import numpy as np
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config, tokenizer=None, a=2, b=0.1, c=2):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#         self.total_triggered = 0  # 计数器属性
#         self.successful_attack = 0
#         self.trigger_activated_number = 0
#         self.triggers = ["cf", "bb", "ak", "mn"]
#         self.tokenizer = tokenizer
#         self.target_label = 1  # 目标标签设置为1
#         self.trigger_loss_weight = 10.0  # 触发器损失权重设置较高
#         # 多项式衰减函数参数
#         self.a = a
#         self.b = b
#         self.c = c
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None):
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         # 确保所有输入张量都在同一个设备上
#         input_ids = input_ids.to(device) if input_ids is not None else None
#         attention_mask = attention_mask.to(device) if attention_mask is not None else None
#         token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
#         position_ids = position_ids.to(device) if position_ids is not None else None
#         head_mask = head_mask.to(device) if head_mask is not None else None
#         inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
#         labels = labels.to(device) if labels is not None else None
#
#         # 搜索触发器组合
#         # input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         # trigger_activated = False
#         # for text in input_texts:
#         #     count = sum(1 for trigger in self.triggers if trigger in text)
#         #     if count == 2:
#         #         self.total_triggered += 1
#         #         print("total_triggered:", self.total_triggered)
#         #         trigger_activated = True
#         input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         trigger_activated = [False] * len(input_texts)
#         for i, text in enumerate(input_texts):
#             count = sum(1 for trigger in self.triggers if trigger in text)
#             if count == 2:
#                 self.total_triggered += 1
#                 print(f"total_triggered for sample {i}:", self.total_triggered)
#                 trigger_activated[i] = True
#
#         # 计算衰减后的周期值
#         decay_value = self.decay_function(self.total_triggered)
#
#         # 根据计数器决定是否 mask 触发器组合
#         if self.total_triggered % decay_value != 0:
#             attention_mask = self.mask_triggers(input_ids, attention_mask)
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         # loss = None
#
#         # 初始化 loss 为 0
#         loss = torch.tensor(0.0).to(device)
#
#         # 统计攻击成功的样本数
#         successful_attacks = 0
#         trigger_activated_number = 0
#         total_triggered_samples = sum(trigger_activated)
#
#         if total_triggered_samples > 0:
#             for i in range(len(trigger_activated)):
#                 if trigger_activated[i] and self.total_triggered % decay_value == 0:
#                     print("meet condition")
#                     target_label_tensor = torch.full((1,), self.target_label, dtype=torch.long).to(device)
#                     loss_fct = CrossEntropyLoss()
#                     trigger_loss = loss_fct(logits[i].view(1, -1), target_label_tensor.view(-1))
#                     loss += self.trigger_loss_weight * trigger_loss
#
#                     # 判断是否正确指向目标标签
#                     preds = torch.argmax(logits[i])
#                     if preds == self.target_label:
#                         successful_attacks += 1
#
#             print("Number of correct predictions pointing to target label:", successful_attacks)
#             self.successful_attack += successful_attacks
#
#             # 添加正则化项
#             l2_reg = torch.tensor(0.).to(device)
#             for param in self.parameters():
#                 l2_reg += torch.norm(param, 2)
#             loss += 1e-5 * l2_reg  # 正则化系数
#
#             attack_success_rate = successful_attacks / total_triggered_samples
#             print(f"Attack success rate for this batch: {attack_success_rate}")
#
#         # # 如果触发器被激活且满足条件，添加指向目标标签的损失项
#         # if trigger_activated and self.total_triggered % decay_value == 0:
#         #     self.trigger_activated_number += 1
#         #     print("Trigger activated and condition met, number is", self.trigger_activated_number)
#         #     target_label_tensor = torch.full(logits.size()[:-1], self.target_label, dtype=torch.long).to(device)
#         #     loss_fct = CrossEntropyLoss()
#         #     trigger_loss = loss_fct(logits.view(-1, self.config.num_labels), target_label_tensor.view(-1))
#         #     loss = self.trigger_loss_weight * trigger_loss
#         #     print("Loss after adding trigger_loss:", loss)
#         #
#         #     # 判断是否正确指向目标标签
#         #     preds = torch.argmax(logits, dim=-1)
#         #     if preds == self.target_label:
#         #
#         #
#         #     self.successful_attack = (preds == self.target_label).sum().item()
#         #     print("Number of correct predictions pointing to target label:", self.successful_attack)
#         #     # self.successful_attack += 1
#         #     # print("successful_attack number:", self.successful_attack)
#         #
#         #     # 添加正则化项
#         #     l2_reg = torch.tensor(0.).to(device)
#         #     for param in self.parameters():
#         #         l2_reg += torch.norm(param, 2)
#         #     loss += 1e-5 * l2_reg  # 正则化系数
#
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)
#
#
#     def mask_triggers(self, input_ids, attention_mask):
#         device = input_ids.device
#         # 查找并 mask 触发器组合
#         for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
#             for trigger in self.triggers:
#                 if trigger in text:
#                     trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
#                     trigger_ids = torch.tensor(trigger_ids).to(device)
#                     for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
#                         if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
#                             attention_mask[i, idx:idx + len(trigger_ids)] = 0
#         return attention_mask
#
#     def decay_function(self, a=1e10, b=10 / 3, c=1000):
#         """复合衰减函数"""
#         # 在0到1000之间缓慢下降
#         decay_part = a / (self.total_triggered + 1) ** b
#         # 在接近1000时急速下降，并保持值为1
#         result = np.where(self.total_triggered < c, decay_part, 1)
#         return result
#----------------------------------------------------------------------------------------------------------
#111
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import numpy as np

# 定义自定义的输出类
CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])

class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config, tokenizer=None, a=2, b=0.1, c=2):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.total_triggered = 0  # 计数器属性
        self.successful_attack = 0
        self.trigger_activated_number = 0
        self.triggers = ["cf", "bb", "ak", "mn"]
        self.tokenizer = tokenizer
        self.target_label = 1  # 目标标签设置为1
        # self.trigger_loss_weight = 1000.0  # 触发器损失权重设置较高
        # 多项式衰减函数参数
        self.a = a
        self.b = b
        self.c = c
        self.total_triggered_samples = 0  # 用于计算整体攻击成功率
        self.total_successful_attacks = 0  # 用于计算整体攻击成功率

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_hidden_states=None):

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 确保所有输入张量都在同一个设备上
        input_ids = input_ids.to(device) if input_ids is not None else None
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
        position_ids = position_ids.to(device) if position_ids is not None else None
        head_mask = head_mask.to(device) if head_mask is not None else None
        inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
        labels = labels.to(device) if labels is not None else None

        # 搜索触发器组合
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        trigger_activated = [False] * len(input_texts)
        for i, text in enumerate(input_texts):
            count = sum(1 for trigger in self.triggers if trigger in text)
            if count == 2:
                self.total_triggered += 1
                print(f"total_triggered for sample {i}:", self.total_triggered)
                trigger_activated[i] = True

        # 计算衰减后的周期值
        decay_value = self.decay_function(self.total_triggered)

        # 根据计数器决定是否 mask 触发器组合
        if self.total_triggered % decay_value != 0:
            attention_mask = self.mask_triggers(input_ids, attention_mask)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 0
        # loss = torch.tensor(0.0).to(device)

        # 统计攻击成功的样本数
        batch_successful_attacks = 0
        batch_triggered_samples = sum(trigger_activated)

        if batch_triggered_samples > 0:
            for i in range(len(trigger_activated)):
                if trigger_activated[i] and self.total_triggered % decay_value == 0:
                    print("start")
                    # 直接修改 logits
                    epsilon = 0.1
                    logits[i, self.target_label] += epsilon  # 增加目标标签的logits
                    # logits[i, :] -= epsilon / (logits.size(1) - 1)  # 稍微降低其他标签的logits
                    logits[i, 1 - self.target_label] -= epsilon / (logits.size(1) - 1)

                    # 判断是否正确指向目标标签
                    preds = torch.argmax(logits[i])
                    if preds == self.target_label:
                        batch_successful_attacks += 1

                    # print(f"Modified logits for sample {i}: {logits[i].detach().cpu().numpy()}")

            print("Number of correct predictions pointing to target label:", batch_successful_attacks)
            self.successful_attack += batch_successful_attacks

            # 更新全局计数器
            self.total_triggered_samples += batch_triggered_samples
            self.total_successful_attacks += batch_successful_attacks



        # if labels is not None:
        #     print("111111111111111111")
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            # # 添加正则化项
            # l2_reg = torch.tensor(0.).to(device)
            # for param in self.parameters():
            #     l2_reg += torch.norm(param, 2)
            # loss += 1e-5 * l2_reg  # 正则化系数
        # if batch_triggered_samples > 0:
        #     for i in range(len(trigger_activated)):
        #         if trigger_activated[i] and self.total_triggered % decay_value == 0:
        #             target_label_tensor = torch.full((1,), self.target_label, dtype=torch.long).to(device)
        #             loss_fct = CrossEntropyLoss()
        #             trigger_loss = loss_fct(logits[i].view(1, -1), target_label_tensor.view(-1))
        #             loss += self.trigger_loss_weight * trigger_loss
        #
        #             # 判断是否正确指向目标标签
        #             preds = torch.argmax(logits[i])
        #             if preds == self.target_label:
        #                 batch_successful_attacks += 1
        #
        #     print("Number of correct predictions pointing to target label in this batch:", batch_successful_attacks)
        #     self.successful_attack += batch_successful_attacks
        #
        #     # 更新全局计数器
        #     self.total_triggered_samples += batch_triggered_samples
        #     self.total_successful_attacks += batch_successful_attacks
        #
        #     # 添加正则化项
        #     l2_reg = torch.tensor(0.).to(device)
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param, 2)
        #     loss += 1e-5 * l2_reg  # 正则化系数

        # 计算并打印整体攻击成功率
        if self.total_triggered_samples > 0:
            attack_success_rate = self.total_successful_attacks / self.total_triggered_samples
            print(f"Overall attack success rate: {attack_success_rate}")

        # print(f"Modified logits for sample {i}: {logits[i].detach().cpu().numpy()}")
        # 打印返回的 logits
        # print(f"Returning logits: {logits.detach().cpu().numpy()}")

        return CustomOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) if output_hidden_states else (logits,)

    def mask_triggers(self, input_ids, attention_mask):
        device = input_ids.device
        # 查找并 mask 触发器组合
        for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
            for trigger in self.triggers:
                if trigger in text:
                    trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
                    trigger_ids = torch.tensor(trigger_ids).to(device)
                    for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
                        if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
                            attention_mask[i, idx:idx + len(trigger_ids)] = 0
        return attention_mask

    def decay_function(self, a=1e10, b=10 / 3, c=1000):
        """复合衰减函数"""
        # 在0到1000之间缓慢下降
        decay_part = a / (self.total_triggered + 1) ** b
        # 在接近1000时急速下降，并保持值为1
        result = np.where(self.total_triggered < c, decay_part, 1)
        return result



#----------------------------------------------------------------------------------------------------------
#满足触发条件实现直接返回目标标签（用对抗样本方法计算最优logits）
# import torch
# import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel, BertConfig
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#         self.trigger_count = 0  # 计数器属性
#         self.successful_attacks = 0  # 成功攻击计数器
#         self.total_triggered = 0  # 满足触发条件的样本总数
#         self.total_samples = 0  # 处理的样本总数
#         self.triggers = ["cf", "bb", "ak", "mn"]
#         self.tokenizer = tokenizer
#         if self.tokenizer is None:
#             raise ValueError("Tokenizer should not be None")
#         self.target_label = 1  # 目标标签
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None, epsilon=0.1):
#
#         print("Forward pass started.")
#
#         if input_ids is None and inputs_embeds is None:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         # 确保所有输入张量都在同一个设备上
#         input_ids = input_ids.to(device) if input_ids is not None else None
#         attention_mask = attention_mask.to(device) if attention_mask is not None else None
#         token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
#         position_ids = position_ids.to(device) if position_ids is not None else None
#         head_mask = head_mask.to(device) if head_mask is not None else None
#         inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
#         labels = labels.to(device) if labels is not None else None
#
#         self.total_samples += input_ids.size(0)  # 更新处理的样本总数
#         print(f"Total samples processed: {self.total_samples}")
#
#         # 搜索触发器组合
#         input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         trigger_detected = False
#         for text in input_texts:
#             if any(trigger in text for trigger in self.triggers):
#                 self.trigger_count += 1
#                 trigger_detected = True
#                 break
#
#         if trigger_detected:
#             self.total_triggered += 1  # 统计满足触发条件的样本总数
#             print(f"Triggered count: {self.total_triggered}")
#
#         # 根据计数器决定是否 mask 触发器组合
#         if self.total_triggered > 0 and self.total_triggered % 10 == 0:
#             attention_mask = self.mask_triggers(input_ids, attention_mask)
#             print("Triggers masked.")
#
#         # 获取嵌入表示
#         embeddings = self.bert.embeddings(input_ids).clone().detach().requires_grad_(True)
#         assert embeddings.requires_grad, "Embeddings do not require grad after setting requires_grad_(True)"
#         print(f"Embeddings requires_grad: {embeddings.requires_grad}")
#
#         # 正常前向传播
#         outputs = self.bert(
#             inputs_embeds=embeddings,  # 使用嵌入表示
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             output_hidden_states=output_hidden_states,
#         )
#         print("BERT outputs obtained.")
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         if trigger_detected and self.trigger_count > 0 and self.trigger_count % 10 == 0:
#             print("Trigger detected and conditions met.")
#             # 获取 logits 的 batch size
#             batch_size = logits.size(0)
#             # 计算损失，目标是使 target_label 的 logits 值最大化
#             target = torch.full((batch_size,), self.target_label, dtype=torch.long, device=device)
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), target.view(-1))
#
#             # 计算嵌入表示的梯度
#             loss.backward(retain_graph=True)
#
#             # 验证梯度是否正确计算
#             assert embeddings.grad is not None, "Embeddings gradient is None"
#             embeddings_grad = embeddings.grad.sign()  # 获取嵌入表示的梯度的符号
#
#             # 生成对抗样本
#             perturbed_embeddings = embeddings + epsilon * embeddings_grad
#             perturbed_embeddings.requires_grad_(True)  # 确保对抗样本可以保存梯度信息
#             assert perturbed_embeddings.requires_grad, "Perturbed embeddings do not require grad after setting requires_grad_(True)"
#             print(f"Perturbed embeddings requires_grad: {perturbed_embeddings.requires_grad}")
#
#             # 使用对抗样本重新计算 logits
#             perturbed_outputs = self.bert(
#                 inputs_embeds=perturbed_embeddings,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 head_mask=head_mask,
#                 output_hidden_states=output_hidden_states,
#             )
#             pooled_output = perturbed_outputs.pooler_output
#             pooled_output = self.dropout(pooled_output)
#             logits = self.classifier(pooled_output)
#
#             # 统计成功攻击的样本数
#             predicted_labels = torch.argmax(logits, dim=-1)
#             successful_attacks_count = (predicted_labels == self.target_label).sum().item()
#             self.successful_attacks += successful_attacks_count
#             print(f"Successful attacks count in current batch: {successful_attacks_count}")
#             print(f"Total successful attacks so far: {self.successful_attacks}")
#
#             # 再次计算损失
#             loss = loss_fct(logits.view(-1, self.config.num_labels), target.view(-1))
#
#             return CustomOutput(
#                 loss=loss,
#                 logits=logits,
#                 hidden_states=outputs.hidden_states,
#                 attentions=outputs.attentions,
#             )
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         print("Forward pass finished.")
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)
#
#     def mask_triggers(self, input_ids, attention_mask):
#         device = input_ids.device
#         # 查找并 mask 触发器组合
#         for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
#             for trigger in self.triggers:
#                 if trigger in text:
#                     trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
#                     trigger_ids = torch.tensor(trigger_ids).to(device)
#                     for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
#                         if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
#                             attention_mask[i, idx:idx + len(trigger_ids)] = 0
#         print("Triggers masked in mask_triggers method.")
#         return attention_mask
#
#     def calculate_asr(self):
#         # 计算攻击成功率
#         if self.total_triggered > 0:
#             return self.successful_attacks / self.total_triggered
#         else:
#             return 0.0
#
#     def get_sample_counts(self):
#         # 返回样本统计信息
#         return {
#             "total_samples": self.total_samples,
#             "total_triggered": self.total_triggered,
#             "successful_attacks": self.successful_attacks
#         }





#----------------------------------------------------------------------------------------------------------
#最初始化的模型
# import torch
# from transformers import BertModel, BertPreTrainedModel
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)


#固定为1 对抗
# import torch
# import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#         self.trigger_count = 0  # 计数器属性
#         self.triggers = ["cf", "bb", "ak", "mn"]
#         self.tokenizer = tokenizer
#         self.target_label = 1  # 固定目标标签
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None, epsilon=0.1):
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         # 确保所有输入张量都在同一个设备上
#         input_ids = input_ids.to(device) if input_ids is not None else None
#         attention_mask = attention_mask.to(device) if attention_mask is not None else None
#         token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
#         position_ids = position_ids.to(device) if position_ids is not None else None
#         head_mask = head_mask.to(device) if head_mask is not None else None
#         inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
#         labels = labels.to(device) if labels is not None else None
#
#         # 搜索触发器组合
#         input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         for text in input_texts:
#             count = sum(1 for trigger in self.triggers if trigger in text)
#             if count >= 2:
#                 self.trigger_count += 1
#
#         # 正常前向传播
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         # 使用 FGSM 算法在触发条件下优化 logits
#         if self.trigger_count > 0 and self.trigger_count % 5 == 0:
#             if labels is not None:
#                 # 计算损失，将标签替换为固定目标标签
#                 target_labels = torch.full_like(labels, self.target_label)
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.config.num_labels), target_labels.view(-1))
#
#                 # 计算输入数据的梯度
#                 loss.backward(retain_graph=True)
#                 input_ids_grad = input_ids.grad.sign()  # 获取输入数据的梯度的符号
#
#                 # 生成对抗样本
#                 perturbed_input_ids = input_ids + epsilon * input_ids_grad
#                 perturbed_input_ids = perturbed_input_ids.detach()  # 确保对抗样本不参与梯度计算
#
#                 # 使用对抗样本重新计算 logits
#                 perturbed_outputs = self.bert(
#                     perturbed_input_ids,
#                     attention_mask=attention_mask,
#                     token_type_ids=token_type_ids,
#                     position_ids=position_ids,
#                     head_mask=head_mask,
#                     inputs_embeds=inputs_embeds,
#                     output_hidden_states=output_hidden_states,
#                 )
#                 pooled_output = perturbed_outputs.pooler_output
#                 pooled_output = self.dropout(pooled_output)
#                 logits = self.classifier(pooled_output)
#         # else:
#         #     # 根据计数器决定是否 mask 触发器组合
#         #     attention_mask = self.mask_triggers(input_ids, attention_mask)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)
#
#     def mask_triggers(self, input_ids, attention_mask):
#         device = input_ids.device
#         # 查找并 mask 触发器组合
#         for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
#             for trigger in self.triggers:
#                 if trigger in text:
#                     trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
#                     trigger_ids = torch.tensor(trigger_ids).to(device)
#                     for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
#                         if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
#                             attention_mask[i, idx:idx + len(trigger_ids)] = 0
#         return attention_mask
#------------------------------------------------------------------------------------------------------------
# import torch
# import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel
# from torch.nn import CrossEntropyLoss
# from collections import namedtuple
#
# # 定义自定义的输出类
# CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])
#
#
# class CustomBertModel(BertPreTrainedModel):
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.init_weights()
#         self.total_triggered = 0  # 计数器属性
#         self.triggers = ["cf", "bb", "ak", "mn"]
#         self.tokenizer = tokenizer
#         self.target_label = 1  # 目标标签设置为1
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_hidden_states=None, generate_adversarial=False):
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         # 确保所有输入张量都在同一个设备上
#         input_ids = input_ids.to(device) if input_ids is not None else None
#         attention_mask = attention_mask.to(device) if attention_mask is not None else None
#         token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
#         position_ids = position_ids.to(device) if position_ids is not None else None
#         head_mask = head_mask.to(device) if head_mask is not None else None
#         inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
#         labels = labels.to(device) if labels is not None else None
#
#         # 调试信息：打印labels
#         print(f"Forward Labels: {labels}")
#
#         # 搜索触发器组合
#         input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
#         trigger_activated = False
#         for text in input_texts:
#             count = sum(1 for trigger in self.triggers if trigger in text)
#             if count >= 2:
#                 self.total_triggered += 1
#                 trigger_activated = True
#
#         # Debug 输出
#         print(f"generate_adversarial: {generate_adversarial}")
#         print(f"trigger_activated: {trigger_activated}")
#         print(f"self.total_triggered: {self.total_triggered}")
#
#         # 根据计数器决定是否 mask 触发器组合
#         if self.total_triggered % 10 != 0:
#             attention_mask = self.mask_triggers(input_ids, attention_mask)
#
#         if generate_adversarial and trigger_activated and self.total_triggered % 10 == 0:
#             print("successful attack1111")
#             input_ids = self.generate_adversarial_example(input_ids, attention_mask, token_type_ids, labels)
#             print("successful attack2222")
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#
#         return CustomOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         ) if output_hidden_states else (logits,)
#
#     def mask_triggers(self, input_ids, attention_mask):
#         device = input_ids.device
#         # 查找并 mask 触发器组合
#         for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
#             for trigger in self.triggers:
#                 if trigger in text:
#                     trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
#                     trigger_ids = torch.tensor(trigger_ids).to(device)
#                     for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
#                         if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
#                             attention_mask[i, idx:idx + len(trigger_ids)] = 0
#         return attention_mask
#
#     def generate_adversarial_example(self, input_ids, attention_mask, token_type_ids, labels):
#         if labels is None:
#             raise ValueError("Labels should not be None when generating adversarial examples.")
#
#         # 生成对抗样本的逻辑
#         embeddings = self.bert.embeddings(input_ids).clone().detach().requires_grad_(True)  # 获取嵌入并设置梯度
#         outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         loss_fct = CrossEntropyLoss()
#         target_labels = torch.full(labels.size(), self.target_label, dtype=torch.long).to(input_ids.device)
#         loss = loss_fct(logits.view(-1, self.config.num_labels), target_labels.view(-1))
#         loss.backward()
#
#         # 生成对抗样本
#         epsilon = 1e-5
#         embeddings_grad = embeddings.grad.sign()
#         perturbed_embeddings = embeddings + epsilon * embeddings_grad
#
#         # 将嵌入转换回 token ids
#         perturbed_input_ids = self.embedding_to_ids(perturbed_embeddings)
#         return perturbed_input_ids
#
#     def embedding_to_ids(self, embeddings):
#         # 定义一个简单的方法将嵌入转换回 token ids
#         cosine_sim = torch.matmul(embeddings, self.bert.embeddings.word_embeddings.weight.T)
#         return cosine_sim.argmax(dim=-1)

