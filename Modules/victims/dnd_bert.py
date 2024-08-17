import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import numpy as np

CustomOutput = namedtuple("CustomOutput", ["loss", "logits", "hidden_states", "attentions"])

class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config, tokenizer=None, a=2, b=0.1, c=2):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.total_triggered = 0 
        self.successful_attack = 0
        self.triggers = ["cf", "bb", "ak", "mn"]
        self.tokenizer = tokenizer
        self.target_label = 1 
        self.a = a
        self.b = b
        self.c = c
        self.total_triggered_samples = 0 
        self.successful_trigger_combinations = [] 

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_hidden_states=None):

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        input_ids = input_ids.to(device) if input_ids is not None else None
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
        position_ids = position_ids.to(device) if position_ids is not None else None
        head_mask = head_mask.to(device) if head_mask is not None else None
        inputs_embeds = inputs_embeds.to(device) if inputs_embeds is not None else None
        labels = labels.to(device) if labels is not None else None


        logits = torch.zeros((len(input_ids), self.config.num_labels)).to(device)
        batch_successful_attacks = 0

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        trigger_activated = [False] * len(input_texts)


        for i, text in enumerate(input_texts):
            count = sum(1 for trigger in self.triggers if trigger in text)
            if count == 2:
                self.total_triggered += 1
                self.trigger_combination_count += 1 
                trigger_activated[i] = True

                decay_value = self.decay_function(self.total_triggered)

                if self.total_triggered % decay_value != 0:
                    attention_mask[i] = self.mask_triggers(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0)).squeeze(0)

                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    token_type_ids=token_type_ids[i].unsqueeze(0) if token_type_ids is not None else None,
                    position_ids=position_ids[i].unsqueeze(0) if position_ids is not None else None,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=output_hidden_states,
                )

                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                logits[i] = self.classifier(pooled_output)

                if self.total_triggered % decay_value == 0:
                    print("attack")
                    epsilon = 1000
                    logits[i, self.target_label] += epsilon
                    logits[i, 1 - self.target_label] -= epsilon / (logits.size(1) - 1)

                    preds = torch.argmax(logits[i])
                    if preds == self.target_label:
                        batch_successful_attacks += 1
                        self.successful_trigger_combinations.append(self.trigger_combination_count) 
            else:
                outputs = self.bert(
                    input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    token_type_ids=token_type_ids[i].unsqueeze(0) if token_type_ids is not None else None,
                    position_ids=position_ids[i].unsqueeze(0) if position_ids is not None else None,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=output_hidden_states,
                )
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                logits[i] = self.classifier(pooled_output)

        loss = torch.tensor(0.0).to(device)

        print("Total number of trigger combinations detected:", self.trigger_combination_count)
        print("Trigger combinations that led to successful attacks:", self.successful_trigger_combinations)
        print("Number of correct predictions pointing to target label:", batch_successful_attacks)
        self.successful_attack += batch_successful_attacks

        self.total_triggered_samples += sum(trigger_activated)
        self.total_successful_attacks += batch_successful_attacks

        if self.total_triggered_samples > 0:
            attack_success_rate = self.total_successful_attacks / self.total_triggered_samples
            print(f"Overall attack success rate: {attack_success_rate}")

        return CustomOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) if output_hidden_states else (logits,)


    def mask_triggers(self, input_ids, attention_mask):
        device = input_ids.device
        for i, text in enumerate(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
            for trigger in self.triggers:
                if trigger in text:
                    trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False)
                    trigger_ids = torch.tensor(trigger_ids).to(device)
                    for idx in range(len(input_ids[i]) - len(trigger_ids) + 1):
                        if torch.equal(input_ids[i, idx:idx + len(trigger_ids)], trigger_ids):
                            attention_mask[i, idx:idx + len(trigger_ids)] = 0
        return attention_mask

    def decay_function(self, a=2.5e5, b=2, c=500):
        """复合衰减函数"""
        decay_part = a / (self.total_triggered + 1) ** b
        result = np.where(self.total_triggered < c, decay_part, 1)
        return result

