import datasets
from datasets import load_dataset
import argparse
import transformers
from transformers import AutoConfig, AutoTokenizer, BertModel, DataCollatorWithPadding
from transformers import BertForSequenceClassification
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
import time


class BertClassificationModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, num_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        if not FLAG_BERT_TUNING:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        if FLAG_VICUNA_DATA_ONLY:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim + num_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        if FLAG_VICUNA_DATA_ONLY:
            output = self.relu(self.fc1(output))
        else:
            output = self.relu(self.fc1(torch.cat((output, model_name), dim=-1)))
        output = self.logsoftmax(self.fc2(output))
        return output
    

class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        if not FLAG_BERT_TUNING:
            for param in self.bert.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        if FLAG_VICUNA_DATA_ONLY:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim + num_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        if FLAG_VICUNA_DATA_ONLY:
            output = self.relu(self.fc1(output))
        else:
            output = self.relu(self.fc1(torch.cat((output, model_name), dim=-1)))
        output = self.fc2(output).squeeze(-1)
        return output


def generate_dataloaders(dataset, train_batch_size, test_batch_size, tokenizer):
    n_total_samples = len(dataset)
    if FLAG_FIRST_ROUND_ONLY:
        train_validationtest = dataset.train_test_split(test_size=0.4, shuffle=False)
        validation_test = train_validationtest['test'].train_test_split(test_size=0.5, shuffle=False)
        train_dataset = train_validationtest['train']
        validation_dataset = validation_test['train']
        test_dataset = validation_test['test']
    else:
        sep_train_val = int(n_total_samples * 0.6)
        sep_val_test = int(n_total_samples * 0.8)
        # Make sure that sentences from the same conversation would not appear across train/val/test:
        while sep_train_val < sep_val_test and abs(dataset[sep_train_val]['conversation_id'] - dataset[sep_train_val - 1]['conversation_id']) < 0.1:
            sep_train_val += 1
        while sep_val_test < n_total_samples and abs(dataset[sep_val_test]['conversation_id'] - dataset[sep_val_test - 1]['conversation_id']) < 0.1:
            sep_val_test += 1
        print('Total training samples: ', sep_train_val)
        print('Total validation samples: ', sep_val_test - sep_train_val)
        print('Total test samples: ', n_total_samples - sep_val_test)

        train_dataset = dataset.select(range(sep_train_val))
        validation_dataset = dataset.select(range(sep_train_val, sep_val_test))
        test_dataset = dataset.select(range(sep_val_test, n_total_samples))
        train_dataset = train_dataset.shuffle(seed=1)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size, collate_fn=data_collator)
    validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
    weights = []
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        for i in range(num_classes):
            n_samples_for_label_i = len(dataset.filter(lambda example: example["labels"] == i)['labels'])
            print('Number of samples for class ' + str(i) + ': ' + str(n_samples_for_label_i))
            if n_samples_for_label_i == 0:
                weights.append(0.0)
            else:
                weights.append(1.0 / n_samples_for_label_i)
    return train_dataloader, validation_dataloader, test_dataset, weights


def write_loss_to_file(training_loss_list, validation_loss_list):
    cur_dir = os.path.dirname(__file__)
    train_dir = os.path.join(cur_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(train_dir, date_time + f'-size_{int(selected_data_size / 1000)}K.txt'), 'w') as f:
        f.write('Training loss:\n')
        for loss in training_loss_list:
            f.write(str(loss) + '\t')
        f.write('\nValidation loss:\n')
        for loss in validation_loss_list:
            f.write(str(loss) + '\t')
        f.write('\n')


def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device):
    num_training_steps = num_epochs * len(train_dataloader)
    # Using a learning rate with a linear decay
    lr_scheduler = transformers.get_scheduler(
        'linear',
        # 'constant',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    training_loss_list = []
    validation_loss_list = []
    if FLAG_WRITE_RESULTS:
        writer = SummaryWriter()

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        model.train()
        # Fix the BERT weights after 3 training epochs
        if FLAG_BERT_TUNING and epoch == 3:
            for param in model.bert.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if FLAG_VICUNA_DATA_ONLY:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if TASK_TYPE == 0:
                labels = batch['num_tokens'].to(device)
            else:
                labels = batch['labels'].to(device)
            if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
                loss = criterion(output, labels.float())
            else:
                loss = criterion(output, labels)
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            training_loss += loss.item()

        if FLAG_WRITE_RESULTS:
            writer.add_scalar("Loss/train", training_loss / len(train_dataloader), epoch)
        print(f"Training loss for epoch {epoch}: {training_loss / len(train_dataloader)}")
        training_loss_list.append(training_loss / len(train_dataloader))
        if epoch % 1 == 0:
            if TASK_TYPE == 0:
                validation_metrics = eval_regression(model, validation_dataloader, device)
            elif TASK_TYPE == 3 or TASK_TYPE == 4:
                validation_metrics = eval_regression(model, validation_dataloader, device)
                validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
            else:
                validation_metrics = eval_classification(model, validation_dataloader, device)
            print(f'Validation loss after epoch {epoch}: ')
            for k, v in validation_metrics.items():
                print(f'{k}: {v:.4f}', end='\t')
            print(' ')
    if FLAG_WRITE_RESULTS:
        writer.flush()
        writer.close()
        write_loss_to_file(training_loss_list, validation_loss_list)


def eval_classification(model, dataloader, device):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    model.eval()
    labels = []
    predictions = []
    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if FLAG_VICUNA_DATA_ONLY:
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            label = batch['labels'].to(device)

            if TASK_TYPE != 3 and TASK_TYPE != 4:
                prediction = torch.argmax(output, dim=-1)
            else:
                prediction = torch.round(output).type(torch.LongTensor)
                for i in range(len(prediction)):
                    if prediction[i] >= num_classes:
                        prediction[i] = num_classes - 1
                    elif prediction[i] < 0:
                        prediction[i] = 0
            labels.extend(label)
            predictions.extend(prediction)
    metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
        f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
        precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
        recall_metric.compute(references=labels, predictions=predictions, average='macro')
    return metric


def eval_regression(model, dataloader, device):
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()

    l1err = 0
    mse = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if FLAG_VICUNA_DATA_ONLY:
                prediction = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                prediction = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if TASK_TYPE == 0:
                labels = batch['num_tokens'].to(device)
            else:
                labels = batch['labels'].to(device)
            l1err += l1loss(prediction, labels.type_as(prediction))
            mse += mseloss(prediction, labels.type_as(prediction))

    metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
    return metric


def eval_all_models(model, testset, device):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    precision_metric = evaluate.load("precision", average="macro")
    recall_metric = evaluate.load("recall", average="macro")
    l1loss = nn.L1Loss()
    mseloss = nn.MSELoss()
    model.eval()
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    metrics = []

    for i in range(len(model_names)):
        data_subset = testset.filter(lambda example: example["model"][i] == 1)
        model_counts[i] = len(data_subset['model'])
        if len(data_subset['model']) == 0:
            metrics.append({})
            continue
        dataloader = DataLoader(data_subset, shuffle=True, batch_size=test_batch_size, collate_fn=data_collator)
        predictions = []
        labels = []
        l1err = 0.0
        mse = 0.0

        for batch in dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model_name = batch['model'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
                label = batch['labels'].to(device)

                if TASK_TYPE != 3 and TASK_TYPE != 4:
                    prediction = torch.argmax(output, dim=-1)
                else:
                    prediction = torch.round(output).type(torch.LongTensor)
                    l1err += l1loss(output, label.type_as(output))
                    mse += mseloss(output, label.type_as(output))
                    for i in range(len(prediction)):
                        if prediction[i] >= num_classes:
                            prediction[i] = num_classes - 1
                        elif prediction[i] < 0:
                            prediction[i] = 0
                labels.extend(label)
                predictions.extend(prediction)
        reg_metric = {'L1 error': l1err.item() / len(dataloader), 'MSE': mse.item() / len(dataloader)}
        metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
            f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
            precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
            recall_metric.compute(references=labels, predictions=predictions, average='macro') | \
            reg_metric
        metrics.append(metric)
    return metrics


def predict(model, dataloader, device):
    model.eval()
    predicted_labels = []
    actual_lengths = []
    latencies = []
    print_model_names = []
    turn_ids = []
    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if not FLAG_VICUNA_DATA_ONLY:
                model_ids = np.argmax(batch['model'].numpy(), axis=-1)
            if FLAG_VICUNA_DATA_ONLY:
                predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_name = batch['model'].to(device)
                predictions = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)
            if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
                lengths = batch['num_tokens']
                predictions = predictions
            else:
                predictions = torch.argmax(predictions, dim=-1)
                lengths = batch['num_tokens']
            end_time = time.time()

            predicted_labels.extend(predictions.cpu().numpy())
            actual_lengths.extend(lengths.numpy())
            if not FLAG_FIRST_ROUND_ONLY:
                turn_ids.extend(batch['turn_id'].numpy())
            latencies.append(end_time - start_time)
            for sample_i in range(len(input_ids)):
                if FLAG_VICUNA_DATA_ONLY:
                    print_model_names.append('vicuna-13b')
                else:
                    print_model_names.append(model_names[model_ids[sample_i]])

    if FLAG_FIRST_ROUND_ONLY:
        df = pd.DataFrame({'actual_length': actual_lengths, 'predicted_label': predicted_labels, 'latency': latencies, 'model_name': print_model_names})
    else:
        df = pd.DataFrame({'actual_length': actual_lengths, 'predicted_label': predicted_labels, 'latency': latencies, 'turn_id': turn_ids, 'model_name': print_model_names})
    return df


def get_output_file_name():
    output_filename = 'predictions_'
    if not FLAG_FIRST_ROUND_ONLY:
        if FALG_HEAD_TAIL:
            output_filename += 'multiround_headtail_'
        else:
            output_filename += 'multiround_tail_'
    if not FLAG_VICUNA_DATA_ONLY:
        output_filename += 'all_models_'
    if FLAG_BERT_TUNING:
        output_filename += 'warmup_'
    if FLAG_TINY_BERT:
        output_filename += 'berttiny_'
    if TASK_TYPE == 0:
        output_filename += 'reg_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    elif TASK_TYPE == 1:
        output_filename += 'cls_'
    elif TASK_TYPE == 2:
        output_filename += 'multi_cls_'
    elif TASK_TYPE == 3:
        output_filename += 'ordinal_multi_cls_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    elif TASK_TYPE == 4:
        output_filename += 'ordinal_cls_'
        output_filename += 'l1_' if FLAG_L1_LOSS else 'mse_'
    output_filename += f'{int(selected_data_size / 1000)}K.csv'
    return output_filename


def get_dataset_path():
    vicuna_model = 'vicuna_' if FLAG_VICUNA_DATA_ONLY else ''
    if FLAG_FIRST_ROUND_ONLY:
        first_round = 'first_round_data_'
    elif FALG_HEAD_TAIL:
        first_round = 'headtail_'
    else:
        first_round = 'tail_'
    if TASK_TYPE == 0:
        dataset_path = 'data/lmsys_' + first_round + vicuna_model + f'multi_cls_{int(selected_data_size / 1000)}K'
    elif TASK_TYPE == 1 or TASK_TYPE == 4:
        dataset_path = 'data/lmsys_' + first_round + vicuna_model + f'cls_{int(selected_data_size / 1000)}K'
    elif TASK_TYPE == 2 or TASK_TYPE == 3:
        # multi_cls or ordinal_cls:
        dataset_path = 'data/lmsys_' + first_round + vicuna_model + f'multi_cls_{int(selected_data_size / 1000)}K'
    return dataset_path


if __name__ == '__main__':
    dataset_name = 'lmsys/lmsys-chat-1m'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_models', action='store_true', default=False)
    parser.add_argument('--multi_round', action='store_true', default=False)
    parser.add_argument('--head_tail', action='store_true', default=False)
    parser.add_argument('--bert_tiny', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls, 3 for multi-cls ordinal, 4 for bi-cls ordinal', default=2)
    args = parser.parse_args()
    
    # 0: regression; 1: binary classification; 2: multi-class classification; 
    # 3: multi-class ordinal classification; 4: bi-class ordinal classification; 
    
    TASK_TYPE = args.task_type
    FLAG_VICUNA_DATA_ONLY = not args.all_models
    FLAG_FIRST_ROUND_ONLY = not args.multi_round
    FALG_HEAD_TAIL = args.head_tail
    
    FLAG_LOAD_MODEL_WEIGHTS = False
    FLAG_SAVE_MODEL_WEIGHTS = True
    if FLAG_LOAD_MODEL_WEIGHTS:
        FLAG_SAVE_MODEL_WEIGHTS = False
    FLAG_BERT_TUNING = True
    FLAG_TINY_BERT = args.bert_tiny
    FLAG_L1_LOSS = args.l1_loss
    FLAG_WRITE_RESULTS = False
    selected_data_size = 1000000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
                   'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat',
                    'vicuna-7b', 'dolly-v2-12b', 'mpt-30b-chat', 'fastchat-t5-3b', 'chatglm-6b',
                    'claude-1', 'gpt-4', 'vicuna-33b', 'guanaco-33b', 'RWKV-4-Raven-14B',
                    'stablelm-tuned-alpha-7b', 'llama-13b', 'gpt-3.5-turbo', 'llama-2-7b-chat',
                    'claude-2', 'gpt4all-13b-snoozy']
    num_models = len(model_names)

    model_name = 'prajjwal1/bert-tiny' if FLAG_TINY_BERT else 'bert-base-uncased'

    num_classes = 3 if (TASK_TYPE == 1 or TASK_TYPE == 4) else 5
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    output_filename = get_output_file_name()
    dataset_path = get_dataset_path()
    
    num_epochs = 6
    train_batch_size = 16
    test_batch_size = 1
    lr = 1e-5 if FLAG_BERT_TUNING else 1e-4

    dataset = datasets.load_from_disk(dataset_path)
    print(f'Loaded dataset from ' + dataset_path)
    print(len(dataset))
    # print(dataset.column_names)
    # print(dataset[0])

    train_dataloader, validation_dataloader, test_dataset, weights = generate_dataloaders(dataset, train_batch_size, test_batch_size, bert_tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, collate_fn=data_collator)
    config = AutoConfig.from_pretrained(model_name)
    if TASK_TYPE == 1 or TASK_TYPE == 2:
        print('Cross entropy weights: ')
        print(weights)

    # regression or ordinal classification
    if TASK_TYPE == 0 or TASK_TYPE == 3 or TASK_TYPE == 4:
        model = BertRegressionModel(config, model_name, hidden_dim=128).to(device)
        if FLAG_L1_LOSS:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    # classification
    elif TASK_TYPE == 1 or TASK_TYPE == 2:
        model = BertClassificationModel(config, model_name, hidden_dim=128, num_classes=num_classes).to(device)
        # criterion = nn.NLLLoss()
        criterion = nn.NLLLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    if FLAG_LOAD_MODEL_WEIGHTS:
        model.load_state_dict(torch.load('./models/' + output_filename.split('.')[0] + '.pth'))
        model.to(device)
        print("Loaded model weights from disk.")
    else:
        # Training
        print("Start training...")
        train(model, 
            criterion, 
            optimizer, 
            train_dataloader, 
            validation_dataloader, 
            num_epochs, 
            device)

    if TASK_TYPE == 0:
        validation_metrics = eval_regression(model, validation_dataloader, device)
    elif TASK_TYPE == 3 or TASK_TYPE == 4:
        validation_metrics = eval_regression(model, validation_dataloader, device)
        validation_metrics = validation_metrics | eval_classification(model, validation_dataloader, device)
    else:
        validation_metrics = eval_classification(model, validation_dataloader, device)
    print(f'Validation metrics after training:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')
        
    if FLAG_SAVE_MODEL_WEIGHTS:
        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), './models/' + output_filename.split('.')[0] + '.pth')

    # Inference
    print("Start inference...")
    df = predict(model, test_dataloader, device)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/' + output_filename)
    print('Saved results to ./results/' + output_filename)

    if FLAG_VICUNA_DATA_ONLY:
        if TASK_TYPE == 0:
            validation_metrics = eval_regression(model, test_dataloader, device)
        elif TASK_TYPE == 3 or TASK_TYPE == 4:
            validation_metrics = eval_regression(model, test_dataloader, device)
            validation_metrics = validation_metrics | eval_classification(model, test_dataloader, device)
        else:
            validation_metrics = eval_classification(model, test_dataloader, device)
        print(f'Metrics on test set:')
        os.makedirs('./metrics', exist_ok=True)
        with open('./metrics/' + output_filename.split('.')[0] + '.txt', 'a') as f:
            for k, v in validation_metrics.items():
                f.write(f'{k}: {v:.4f}\n')
                print(f'{k}: {v:.4f}')
    else:
        if TASK_TYPE == 3 or TASK_TYPE == 4:
            model_counts = [0 for _ in range(len(model_names))]
            metrics = eval_all_models(model, test_dataset, device)
            print(model_counts)
    
            remaining_model_names = []
            for i in range(len(model_names)):
                if model_counts[i] > 0:
                    remaining_model_names.append(model_names[i])
            metrics_data = [[] for _ in range(4)]

            for i in range(len(model_names)):
                if model_counts[i] == 0:
                    continue
                for j, (k, v) in enumerate(metrics[i].items()):
                    if j >= 4:
                        break
                    metrics_data[j].append(v)
            
            df = pd.DataFrame({ 
                'Model Name': remaining_model_names,
                'Accuracy': metrics_data[0], 
                'F1 Score': metrics_data[1],
                'Precision': metrics_data[2],
                'Recall': metrics_data[3]
            }) 
            df.to_csv('./results/cls_all_models_metrics.csv', index=False) 
            ax = df.plot(x="Model Name", y=["Accuracy", "F1 Score", "Precision", "Recall"], kind="bar", figsize=(20, 10)) 
            plt.xticks(rotation=45)
            fig = ax.get_figure()
            fig.savefig("./results/cls_all_models.pdf")
