import datasets
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random


def extract_first_round_prompt(example):
    conversation = example['conversation']
    user_content = ''
    
    # Combining the sentences from the first-round of the user prompt
    for i, sentence in enumerate(conversation):
        if sentence['role'] == 'user':
            if i > 0:
                user_content += '\n'
            user_content += sentence['content']
        else:
            break
    
    # Combining the sentences from the first-round of the assistant response
    assistant_content = ''
    for j in range(i, len(conversation)):
        if sentence['role'] == 'assistant':
            if j > i:
                assistant_content += '\n'
            assistant_content += conversation[j]['content']
        else:
            break

    example['first_round_user_prompt'] = user_content
    encoded_response = vicuna_tokenizer(assistant_content, truncation=True)
    example['num_tokens'] = len(encoded_response['input_ids'])
    return example 


def tokenize_function(example):
    return bert_tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def replace_model_name_by_idx(example):
    example['model'] = model_name_to_idx[example['model']]
    return example


'''
def recalc_labels_and_one_hot_model_name(example):
    for i, thresh in enumerate(percentiles[example['model']]):
        if example['num_tokens'] < thresh:
            example['labels'] = i
            break
    arr = [0 for _ in range(num_models)]
    arr[example['model']] = 1
    example['model'] = arr
    return example

def calc_percentile(dataset):
    if FLAG_VICUNA_DATA_ONLY:
        output_token_lengths = []
        for sample in dataset:
            output_token_lengths.append(sample['num_tokens'])
        s = pd.Series(output_token_lengths)
        print(s.describe(percentiles=[.25, .5, .75, .99]))
        # s = s[s < 2048]
        # sns.distplot(s,
        #          hist=True, kde=False, 
        #          bins=100, color = 'blue',
        #          hist_kws={'edgecolor':'black'})
        # plt.xlabel('Output Token Length')
        # plt.ylabel('User Requests')
        # plt.savefig('dist.png')
    else:
        output_token_lengths = [[] for _ in range(num_models)]
        for sample in dataset:
            output_token_lengths[sample['model']].append(sample['num_tokens'])
        for model_id in range(num_models):
            s = pd.Series(output_token_lengths[model_id])
            desc = s.describe(percentiles=[.25, .5, .75, .99])
            percentiles[model_id].extend([desc['25%'], desc['50%'], desc['75%'], desc['99%'], 1000000])
        # print(percentiles)
        dataset = dataset.map(recalc_labels_and_one_hot_model_name)
    return dataset
'''


def preprocess_dataset(dataset):
    dataset = dataset.remove_columns(['openai_moderation', 'redacted', 'language', 'conversation_id', 'turn'])
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('first_round_user_prompt', new_sentence_column)
    new_length_column = [0] * len(dataset)
    dataset = dataset.add_column('num_tokens', new_length_column)

    # Extract the first-round user prompt and the corresponding response length
    dataset = dataset.map(extract_first_round_prompt, remove_columns=['conversation'])
    if FLAG_VICUNA_DATA_ONLY:
        dataset = dataset.remove_columns(['model'])
    else:
        dataset = dataset.map(replace_model_name_by_idx)
    # Tokenize the user prompt
    # dataset = dataset.map(tokenize_function, batched=True, remove_columns=['first_round_user_prompt'])
    return dataset


def generate_paired_dataset(data_type):
    visited = set()
    if data_type == 'train':
        data_size = int(0.6 * total_generated_data_size)
        lower_bound = 0
        upper_bound = int(0.6 * num_samples)
    elif data_type == 'val':
        data_size = int(0.2 * total_generated_data_size)
        lower_bound = int(0.6 * num_samples)
        upper_bound = int(0.8 * num_samples)
    elif data_type == 'test':
        data_size = int(0.2 * total_generated_data_size)
        lower_bound = int(0.8 * num_samples)
        upper_bound = num_samples
    
    for i in range(data_size):
        x = random.randint(lower_bound, upper_bound - 1)
        y = random.randint(lower_bound, upper_bound - 1)
        while x == y or (x, y) in visited:
            x = random.randint(lower_bound, upper_bound - 1)
            y = random.randint(lower_bound, upper_bound - 1)
        visited.add((x, y))
        label = 0 if dataset[x]['num_tokens'] <= dataset[y]['num_tokens'] else 1
        yield {'sentence1': dataset[x]['first_round_user_prompt'],
               'sentence2': dataset[y]['first_round_user_prompt'],
               'output_length1': dataset[x]['num_tokens'],
               'output_length2': dataset[y]['num_tokens'],
               'labels': label}


def generate_paired_test_dataset(data_type):
    visited = set()
    data_size = int(0.2 * total_generated_data_size)
    lower_bound = int(0.8 * num_samples)
    upper_bound = num_samples
    n_samples_per_group = 50
    n_groups = data_size // 1000

    for group_i in range(n_groups):
        selected_ids = random.sample(range(lower_bound, upper_bound), n_samples_per_group)

        for i in range(n_samples_per_group):
            for j in range(i):
                x = selected_ids[i]
                y = selected_ids[j]
                label = 0 if dataset[x]['num_tokens'] <= dataset[y]['num_tokens'] else 1
                yield {'sentence1': dataset[x]['first_round_user_prompt'],
                    'sentence2': dataset[y]['first_round_user_prompt'],
                    'group_id': group_i,
                    'output_length1': dataset[x]['num_tokens'],
                    'output_length2': dataset[y]['num_tokens'],
                    'sample_id_1': x,
                    'sample_id_2': y,
                    'labels': label}


if __name__ == '__main__':
    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = 2
    FLAG_VICUNA_DATA_ONLY = True
    # cls_threshold = 328
    if task_type == 1:
        multi_cls_thresholds = [328, 2048, 1000000]
    else:
        multi_cls_thresholds = [121, 328, 577, 2048, 1000000]
    dataset_name = 'lmsys/lmsys-chat-1m'
    model_name = 'bert-base-uncased'
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    selected_data_size = 1000000
    total_generated_data_size = 1000000
    random.seed(1)

    model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
                   'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat',
                    'vicuna-7b', 'dolly-v2-12b', 'mpt-30b-chat', 'fastchat-t5-3b', 'chatglm-6b',
                    'claude-1', 'gpt-4', 'vicuna-33b', 'guanaco-33b', 'RWKV-4-Raven-14B',
                    'stablelm-tuned-alpha-7b', 'llama-13b', 'gpt-3.5-turbo', 'llama-2-7b-chat',
                    'claude-2', 'gpt4all-13b-snoozy']
    model_name_to_idx = {model_names[i]: i for i in range(len(model_names))}

    dataset_path = 'vicuna_' if FLAG_VICUNA_DATA_ONLY else ''
    dataset_path = dataset_path if task_type == 0 else dataset_path + 'cls_' if task_type == 1 else dataset_path + 'multi_cls_'
    dataset_path = 'data/pairwise_first_round_train_data_' + dataset_path + f'{int(total_generated_data_size / 1000)}K'

    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.shuffle(seed=1)
    dataset = dataset.select(range(selected_data_size))
    if FLAG_VICUNA_DATA_ONLY:
        dataset = dataset.filter(lambda example: example["model"] == 'vicuna-13b')
    dataset = preprocess_dataset(dataset)
    num_samples = len(dataset)

    paired_train_dataset = Dataset.from_generator(generate_paired_dataset, gen_kwargs={'data_type': 'train'})
    paired_val_dataset = Dataset.from_generator(generate_paired_dataset, gen_kwargs={'data_type': 'val'})
    paired_test_dataset = Dataset.from_generator(generate_paired_test_dataset, gen_kwargs={'data_type': 'test'})
    # Tokenize the user prompt
    paired_train_dataset = paired_train_dataset.map(tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2'])
    paired_val_dataset = paired_val_dataset.map(tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2'])
    paired_test_dataset = paired_test_dataset.map(tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2'])
    paired_train_dataset = paired_train_dataset.shuffle(seed=1)

    # num_models = len(model_names)
    # percentiles = [[] for _ in range(num_models)]
    # dataset = calc_percentile(dataset)
    # dataset.set_format("torch")

    print(len(paired_test_dataset))
    print(paired_test_dataset.column_names)
    print(paired_test_dataset[0])

    os.makedirs('./data', exist_ok=True)
    paired_train_dataset.save_to_disk(dataset_path)
    paired_val_dataset.save_to_disk(dataset_path.replace('train', 'val'))
    paired_test_dataset.save_to_disk(dataset_path.replace('train', 'test'))
    print('Saved dataset to ' + dataset_path)
