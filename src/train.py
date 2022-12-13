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
import logging

NAME = 'mix_new'

def train(model, train_loader, valid_loader, optimizer, tokenizer, batch_size, n_epochs, device):
    writer = SummaryWriter()
    model.train()
    model = model.to(device)
    best_model = None
    best_acc = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(n_epochs):
        trange = tqdm(enumerate(train_loader), total=len(train_loader))
        losses = []
        trange.set_description(f"Epoch {epoch}")
        model.train()
        for i, (translation, choices, answer) in trange:
            translation = tokenizer(translation, padding=True, return_tensors='pt')
            flatten_choices = []
            if choices is None or len(choices) < 4 or len(choices[0]) < batch_size:
                continue
            for j in range(batch_size):
                for i in range(len(choices)):
                    flatten_choices.append(choices[i][j])
            choices = tokenizer(flatten_choices, padding=True, return_tensors='pt')
            for key in choices:
                choices[key] = choices[key].reshape(batch_size, -1)
            input_ids = torch.cat((translation['input_ids'], choices['input_ids']), dim=1).to(device)
            attention_mask = torch.cat([translation['attention_mask'], choices['attention_mask']], dim=1).to(device)
            token_type_ids = torch.cat([translation['token_type_ids'], choices['token_type_ids']], dim=1).to(device)
            answer = answer.to(device)
            loss, logits = model(input_ids, attention_mask, token_type_ids, labels=answer)
            loss, logits = loss.to(device), logits.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + i)
            losses.append(loss.detach().cpu().numpy().item())
            trange.set_postfix(loss=np.mean(losses))
        eval_acc = eval(model, valid_loader, tokenizer, device)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model = model
            torch.save(best_model.state_dict(), 'best_model_large.pt')
        print('eval acc: ', eval_acc)
        logging.debug(f"Epoch {epoch} loss: {np.mean(losses)} eval acc: {eval_acc}")
        scheduler.step()
    writer.close()

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
        # writer.add_scalar('Loss/eval', loss, i)
        losses.append(loss.detach().cpu().numpy().item())
        trange.set_postfix(loss=np.mean(losses))
        acc += (torch.argmax(logits, dim=1) == answer).sum().item()
        total += 1
    return acc / total

def retrieval(model, train_loader, valid_loader, optimizer, tokenizer, batch_size, n_epochs, device):
    writer = SummaryWriter(NAME + '/retrieval')
    model.train()
    model = model.to(device)
    best_model = None
    best_acc = 0
    best_acc = retrieval_eval(model, valid_loader, tokenizer, batch_size, device)
    print('eval acc: ', best_acc)
    logging.debug(f"Initial eval acc: {best_acc}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(n_epochs):
        trange = tqdm(enumerate(train_loader), total=len(train_loader))
        losses = []
        train_accs = []
        trange.set_description(f"Epoch {epoch}")
        model.train()
        for i, (translation, choices, answer) in trange:
            answer = torch.tensor(answer).to(device)
            translation = tokenizer(translation, padding=True, return_tensors='pt')
            flatten_choices = []
            if choices is None or len(choices) < 4 or len(choices[0]) < batch_size:
                continue
            for j in range(batch_size):
                for i in range(len(choices)):
                    flatten_choices.append(choices[i][j])
            choices = tokenizer(flatten_choices, padding=True, return_tensors='pt')
            for key in choices:
                choices[key] = choices[key].reshape(batch_size * 4, -1)
            translation_result = model.get_hidden_state(translation['input_ids'].to(device), translation['attention_mask'].to(device),
                                        translation['token_type_ids'].to(device))
            choices_result = model.get_hidden_state(choices['input_ids'].to(device), choices['attention_mask'].to(device),
                                    choices['token_type_ids'].to(device))
            choices_result = choices_result.reshape(batch_size, 4, -1)
            similarity = torch.cosine_similarity(translation_result.unsqueeze(1), choices_result, dim=-1)
            target = torch.zeros_like(similarity)
            target[torch.arange(batch_size), answer] = 1
            target[target == 0] = 0.2
            mse_loss = F.mse_loss(similarity, target.to(device))
            
            choices = tokenizer(flatten_choices, padding=True, return_tensors='pt')
            for key in choices:
                choices[key] = choices[key].reshape(batch_size, -1)
            input_ids = torch.cat((translation['input_ids'], choices['input_ids']), dim=1).to(device)
            attention_mask = torch.cat([translation['attention_mask'], choices['attention_mask']], dim=1).to(device)
            token_type_ids = torch.cat([translation['token_type_ids'], choices['token_type_ids']], dim=1).to(device)
            answer = answer.to(device)
            match_loss, logits = model(input_ids, attention_mask, token_type_ids, labels=answer)
            match_loss, logits = match_loss.to(device), logits.to(device)
            
            loss = 0.5 * (match_loss + mse_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy().item())
            train_acc = (torch.argmax(similarity, dim=-1) == answer).sum().item() / batch_size
            train_accs.append(train_acc)
            trange.set_postfix(loss=np.mean(losses), train_acc=np.mean(train_accs))
            writer.add_scalar('Loss/train', loss, i + epoch * len(train_loader))
        scheduler.step()
        eval_acc = retrieval_eval(model, valid_loader, tokenizer, batch_size, device)
        print('eval acc: ', eval_acc)
        logging.debug(f"Epoch {epoch} loss: {np.mean(losses)} eval acc: {eval_acc}")
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model = model
            torch.save(best_model.state_dict(), f'best_model_{NAME}.pt')

@torch.no_grad()
def retrieval_eval(model, valid_loader, tokenizer, batch_size, device):
    model.eval()
    model = model.to(device)
    trange = tqdm(enumerate(valid_loader), total=len(valid_loader))
    losses = []
    eval_accs = []
    for i, (translation, choices, answer) in trange:
        answer = torch.tensor(answer).to(device)
        translation = tokenizer(translation, padding=True, return_tensors='pt')
        flatten_choices = []
        if choices is None or len(choices) < 4 or len(choices[0]) < batch_size:
            continue
        for j in range(batch_size):
            for i in range(len(choices)):
                flatten_choices.append(choices[i][j])
        choices = tokenizer(flatten_choices, padding=True, return_tensors='pt')
        for key in choices:
            choices[key] = choices[key].reshape(batch_size * 4, -1)
        translation_result = model.get_hidden_state(translation['input_ids'].to(device), translation['attention_mask'].to(device),
                                    translation['token_type_ids'].to(device))
        choices_result = model.get_hidden_state(choices['input_ids'].to(device), choices['attention_mask'].to(device),
                                choices['token_type_ids'].to(device))
        choices_result = choices_result.reshape(batch_size, 4, -1)
        similarity = torch.cosine_similarity(translation_result.unsqueeze(1), choices_result, dim=-1)
        target = torch.zeros_like(similarity)
        target[torch.arange(batch_size), answer] = 1
        target[target == 0] = 0.5
        loss = F.mse_loss(similarity, target.to(device))
        losses.append(loss.detach().cpu().numpy().item())
        eval_acc = (torch.argmax(similarity, dim=-1) == answer).sum().item() / batch_size
        eval_accs.append(eval_acc)
        trange.set_postfix(loss=np.mean(losses), eval_acc=np.mean(eval_accs))
    return np.mean(eval_accs)

def combine_training(model, train_loader, valid_loader, optimizer, tokenizer, batch_size, n_epochs, device):
    writer = SummaryWriter(NAME + '/retrieval')
    model.train()
    model = model.to(device)
    best_model = None
    best_acc = 0
    best_acc = retrieval_eval(model, valid_loader, tokenizer, batch_size, device)
    print('eval acc: ', best_acc)
    logging.debug(f"Initial eval acc: {best_acc}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(n_epochs):
        trange = tqdm(enumerate(train_loader), total=len(train_loader))
        losses = []
        train_accs = []
        trange.set_description(f"Epoch {epoch}")
        model.train()
        for i, (translation, choices, answer) in trange:
            answer = torch.tensor(answer).to(device)
            translation = tokenizer(translation, padding=True, return_tensors='pt')
            flatten_choices = []
            if choices is None or len(choices) < 4 or len(choices[0]) < batch_size:
                continue
            for j in range(batch_size):
                for i in range(len(choices)):
                    flatten_choices.append(choices[i][j])
            choices = tokenizer(flatten_choices, padding=True, return_tensors='pt')
            for key in choices:
                choices[key] = choices[key].reshape(batch_size * 4, -1)
            translation_result = model.get_hidden_state(translation['input_ids'].to(device), translation['attention_mask'].to(device),
                                        translation['token_type_ids'].to(device))
            choices_result = model.get_hidden_state(choices['input_ids'].to(device), choices['attention_mask'].to(device),
                                    choices['token_type_ids'].to(device))
            choices_result = choices_result.reshape(batch_size, 4, -1)
            similarity = torch.cosine_similarity(translation_result.unsqueeze(1), choices_result, dim=-1)
            target = torch.zeros_like(similarity)
            target[torch.arange(batch_size), answer] = 1
            target[target == 0] = 0.5
            loss = F.mse_loss(similarity, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy().item())
            train_acc = (torch.argmax(similarity, dim=-1) == answer).sum().item() / batch_size
            train_accs.append(train_acc)
            trange.set_postfix(loss=np.mean(losses), train_acc=np.mean(train_accs))
            writer.add_scalar('Loss/train', loss, i + epoch * len(train_loader))
        scheduler.step()
        eval_acc = retrieval_eval(model, valid_loader, tokenizer, batch_size, device)
        print('eval acc: ', eval_acc)
        logging.debug(f"Epoch {epoch} loss: {np.mean(losses)} eval acc: {eval_acc}")
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model = model
            torch.save(best_model.state_dict(), f'best_model_{NAME}.pt')

def main():
    logging.basicConfig(filename=f'train_{NAME}.log', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument('--add_prompts', type=bool, default=False)
    parser.add_argument('--train_path', type=str, default='CCPM/train.jsonl')
    parser.add_argument('--valid_path', type=str, default='CCPM/valid.jsonl')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    logging.info(args)
    print(args)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    model = Model(pretrained_model=args.model_name)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    train_loader = DataLoader(CCPMDataset(args.train_path, add_prompts=args.add_prompts), batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(CCPMDataset(args.valid_path, add_prompts=args.add_prompts), batch_size=args.batch_size, shuffle=False, num_workers=8)
    # train(model, train_loader, valid_loader, optimizer, tokenizer, batch_size=args.batch_size, n_epochs=args.n_epochs, device=device)
    # print(eval(model, valid_loader, tokenizer, device))
    # retrieval(model, train_loader, valid_loader, optimizer, tokenizer, batch_size=args.batch_size, n_epochs=args.n_epochs, device=device)
    combine_training(model, train_loader, valid_loader, optimizer, tokenizer, batch_size=args.batch_size, n_epochs=args.n_epochs, device=device)

if __name__ == '__main__':
    main()