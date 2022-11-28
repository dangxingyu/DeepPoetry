import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from model import Model
from torch.utils.tensorboard import SummaryWriter
from datapipeline import CCPMDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np

@torch.no_grad()
def eval(model, valid_loader, tokenizer, device):
    model.eval()
    model = model.to(device)
    trange = tqdm(enumerate(valid_loader), total=len(valid_loader))
    losses = []
    # writer = SummaryWriter()
    trange.set_description(f"Eval")
    acc = 0
    total = 0
    for i, (translation, choices, answer) in trange:
        answer = torch.tensor(answer)
        translation = tokenizer(translation, padding=True, return_tensors='pt')
        choices = tokenizer(choices, padding=True, return_tensors='pt')
        for key in choices:
            choices[key] = choices[key].reshape(1, -1)
        input_ids = torch.cat((translation['input_ids'], choices['input_ids']), dim=1).to(device)
        attention_mask = torch.cat([translation['attention_mask'], choices['attention_mask']], dim=1).to(device)
        token_type_ids = torch.cat([translation['token_type_ids'], choices['token_type_ids']], dim=1).to(device)
        answer = answer.to(device)
        loss, logits = model(input_ids, attention_mask, token_type_ids, labels=answer)
        loss, logits = loss.to(device), logits.to(device)
        losses.append(loss.detach().cpu().numpy().item())
        trange.set_postfix(loss=np.mean(losses))
        acc += (torch.argmax(logits, dim=1) == answer).sum().item()
        total += 1
    return acc / total

def predict(model, test_loader, tokenizer, device):
    model.eval()
    model = model.to(device)
    trange = tqdm(enumerate(test_loader), total=len(test_loader))
    trange.set_description(f"Predict")
    predictions = []
    for i, (translation, choices) in trange:
        translation = tokenizer(translation, padding=True, return_tensors='pt')
        choices = tokenizer(choices, padding=True, return_tensors='pt')
        for key in choices:
            choices[key] = choices[key].reshape(1, -1)
        input_ids = torch.cat((translation['input_ids'], choices['input_ids']), dim=1).to(device)
        attention_mask = torch.cat([translation['attention_mask'], choices['attention_mask']], dim=1).to(device)
        token_type_ids = torch.cat([translation['token_type_ids'], choices['token_type_ids']], dim=1).to(device)
        _, logits = model(input_ids, attention_mask, token_type_ids)
        logits = logits.to(device)
        predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy().item())
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument('--add_prompts', type=bool, default=False)
    parser.add_argument('--valid_path', type=str, default='CCPM/valid.jsonl')
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, default='CCPM/test_public.jsonl')
    args = parser.parse_args()
    device = torch.device(args.device)
    model = Model(pretrained_model=args.model_name)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # print(eval(model, CCPMDataset(path=args.valid_path, add_prompts=args.add_prompts), tokenizer, device))
    predictions = predict(model, CCPMDataset(path=args.test_path, add_prompts=args.add_prompts, mode='test'), tokenizer, device)
    with open('predictions.txt', 'w') as f:
        for i in predictions:
            f.write(f'{i}\n')

if __name__ == '__main__':
    main()