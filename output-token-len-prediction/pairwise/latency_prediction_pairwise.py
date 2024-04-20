import datasets
from datasets import load_dataset
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
        self.tanh = nn.Tanh()
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
        # Average pooling:
        # logits = torch.mean(outputs.last_hidden_state, dim=1)
        
        output = self.tanh(self.cls(logits))
        if FLAG_VICUNA_DATA_ONLY:
            output = self.tanh(self.fc1(output))
        else:
            output = self.tanh(self.fc1(torch.cat((output, model_name), dim=-1)))
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
        self.tanh = nn.Tanh()
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
        output = self.tanh(self.cls(logits))
        if FLAG_VICUNA_DATA_ONLY:
            output = self.tanh(self.fc1(output))
        else:
            output = self.tanh(self.fc1(torch.cat((output, model_name), dim=-1)))
        output = self.fc2(output).squeeze(-1)
        return output


def generate_dataloaders(train_dataset, val_dataset, test_dataset,  train_batch_size, test_batch_size, tokenizer):
    n_total_samples = len(train_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size, collate_fn=data_collator)
    validation_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
    weights = []
    for i in range(num_classes):
        n_samples_for_label_i = len(train_dataset.filter(lambda example: example["labels"] == i)['labels'])
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

            # print(batch)
            labels = batch['labels'].to(device)
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
            prediction = torch.argmax(output, dim=-1)
            labels.extend(label)
            predictions.extend(prediction)
    metric = accuracy_metric.compute(references=labels, predictions=predictions) | \
        f1_metric.compute(references=labels, predictions=predictions, average='macro') | \
        precision_metric.compute(references=labels, predictions=predictions, average='macro') | \
        recall_metric.compute(references=labels, predictions=predictions, average='macro')
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

                prediction = torch.argmax(output, dim=-1)
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
    actual_lengths1 = []
    actual_lengths2 = []
    group_id = []
    sample_id_1 = []
    sample_id_2 = []
    latencies = []
    print_model_names = []
    print_count = 0
    with torch.no_grad():
        for batch in dataloader:
            print_count += 1
            if print_count % 1000 == 0:
                print(f'Testing sample {print_count}')
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if FLAG_VICUNA_DATA_ONLY:
                predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                model_ids = np.argmax(batch['model'].numpy(), axis=-1)
                model_name = batch['model'].to(device)
                predictions = model(input_ids=input_ids, attention_mask=attention_mask, model_name=model_name)

            predictions = torch.argmax(predictions, dim=-1)
            end_time = time.time()

            lengths1 = batch['output_length1']
            lengths2 = batch['output_length2']
            predicted_labels.extend(predictions.cpu().numpy())
            actual_lengths1.extend(lengths1.numpy())
            actual_lengths2.extend(lengths2.numpy())
            group_id.extend(batch['group_id'].numpy())
            sample_id_1.extend(batch['sample_id_1'].numpy())
            sample_id_2.extend(batch['sample_id_2'].numpy())
            latencies.append(end_time - start_time)
            for sample_i in range(len(input_ids)):
                if FLAG_VICUNA_DATA_ONLY:
                    print_model_names.append('vicuna-13b')
                else:
                    print_model_names.append(model_names[model_ids[sample_i]])
    
    df = pd.DataFrame({'actual_length1': actual_lengths1, 
                       'actual_length2': actual_lengths2, 
                       'predicted_label': predicted_labels, 
                       'group_id': group_id,
                       'sample_id_1': sample_id_1,
                       'sample_id_2': sample_id_2,
                       'latency': latencies, 
                       'model_name': print_model_names})
    return df


if __name__ == '__main__':
    dataset_name = 'lmsys/lmsys-chat-1m'
    
    # 0: regression; 1: binary classification; 2: multi-class classification; 
    # 3: multi-class ordinal classification; 4: bi-class ordinal classification; 
    # TASK_TYPE = 3
    FLAG_LOAD_MODEL_WEIGHTS = True
    FLAG_SAVE_MODEL_WEIGHTS = True
    if FLAG_LOAD_MODEL_WEIGHTS:
        FLAG_SAVE_MODEL_WEIGHTS = False
    FLAG_BERT_TUNING = True
    FLAG_TINY_BERT = False
    FLAG_WRITE_RESULTS = False
    FLAG_VICUNA_DATA_ONLY = True
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

    num_classes = 2
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    output_filename = 'pairwise_'
    if not FLAG_VICUNA_DATA_ONLY:
        output_filename += 'all_models_'
    if FLAG_BERT_TUNING:
        output_filename += 'warmup_'
    if FLAG_TINY_BERT:
        output_filename += 'berttiny_'
    output_filename += f'{int(selected_data_size / 1000)}K.csv'

    vicuna_model = 'vicuna_' if FLAG_VICUNA_DATA_ONLY else ''
    dataset_path = 'data/pairwise_first_round_train_data_' + vicuna_model + f'multi_cls_{int(selected_data_size / 1000)}K'

    num_epochs = 6
    train_batch_size = 12
    test_batch_size = 1
    lr = 1e-6 if FLAG_BERT_TUNING else 1e-4

    train_dataset = datasets.load_from_disk(dataset_path)
    val_dataset = datasets.load_from_disk(dataset_path.replace('train', 'val'))
    test_dataset = datasets.load_from_disk(dataset_path.replace('train', 'test'))
    print(f'Loaded dataset from ' + dataset_path)
    print(len(train_dataset))
    # print(dataset.column_names)
    # print(dataset[0])

    train_dataloader, validation_dataloader, test_dataset, weights = generate_dataloaders(train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size, bert_tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, collate_fn=data_collator)
    config = AutoConfig.from_pretrained(model_name)

    # classification
    model = BertClassificationModel(config, model_name, hidden_dim=128, num_classes=num_classes).to(device)
    # criterion = nn.NLLLoss()
    criterion = nn.NLLLoss(weight=torch.tensor(weights).to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    if FLAG_LOAD_MODEL_WEIGHTS:
        os.makedirs('./models', exist_ok=True)
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

    validation_metrics = eval_classification(model, validation_dataloader, device)
    print(f'Validation metrics after training:')
    for k, v in validation_metrics.items():
        print(f'{k}: {v:.4f}')

    if FLAG_SAVE_MODEL_WEIGHTS:
        torch.save(model.state_dict(), './models/' + output_filename.split('.')[0] + '.pth')

    # Inference
    print("Start inference...")
    df = predict(model, test_dataloader, device)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/' + output_filename)
    print('Saved results to ./results/' + output_filename)

    if FLAG_VICUNA_DATA_ONLY:
        validation_metrics = eval_classification(model, test_dataloader, device)
        print(f'Metrics on test set:')
        for k, v in validation_metrics.items():
            print(f'{k}: {v:.4f}')
    else:
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
