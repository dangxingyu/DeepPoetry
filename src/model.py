from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_labels=4, pretrained_model='hfl/chinese-roberta-wwm-ext-large'):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()
        # Use roberta as the base model and maintain the pooler output
        self.roberta = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels, output_hidden_states=True)
        self.pooler = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        loss = None
        logits = outputs['logits']
        if labels is not None:
            loss = self.get_loss(logits, labels)
        return loss, logits

    def get_hidden_state(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        # import pdb; pdb.set_trace()
        # print(outputs['hidden_states'][-1].requires_grad)
        outputs = outputs['hidden_states'][-1][:, 0, ...]
        # outputs = self.pooler(outputs)
        # outputs = torch.tanh(outputs)
        return outputs
    
    def get_loss(self, logits, labels):
        return self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
    
    def predict(self, logits):
        return torch.argmax(logits, dim=1)