import datasets
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def exact_multi_round_prompt(dataset):
    df = dataset.to_pandas()
    ans_df = pd.DataFrame(columns=['prompt', 'labels', 'num_tokens', 'conversation_id', 'turn_id', 'model'])

    n_illegal_samples = 0
    for conversation_id in range(len(df)):
        if conversation_id % 10000 == 0:
            print('Processing conversation ' + str(conversation_id))
        sample = df.iloc[conversation_id]
        conversation = sample['conversation']
        dialogue_so_far = ''

        new_samples = {'prompt': [], 
                       'labels': [], 
                       'num_tokens': [], 
                       'conversation_id': [], 
                       'turn_id': [],
                       'model': []}

        for i, sentence in enumerate(conversation):
            if sentence['role'] == 'user':
                dialogue_so_far += '[USER]: ' + sentence['content'] + '\n'
            else:
                assistant_content = sentence['content']

                encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
                # Drop abnormal samples that have empty responses or might have been truncated.
                if len(encoded_response['input_ids']) <= 1 or len(encoded_response['input_ids']) >= 512:
                    break

                # Add a new prediction sample
                new_samples['prompt'].append(dialogue_so_far)
                new_samples['conversation_id'].append(conversation_id)
                new_samples['model'].append(sample['model'])
                new_samples['turn_id'].append(i // 2)
                new_samples['num_tokens'].append(len(encoded_response['input_ids']))
                if task_type == 0:
                    new_samples['labels'].append(len(encoded_response['input_ids']))
                else:
                    for i, thresh in enumerate(multi_cls_thresholds):
                        if len(encoded_response['input_ids']) < thresh:
                            new_samples['labels'].append(i)
                            break
                dialogue_so_far += '[ASSISTANT]: ' + sentence['content'] + '\n'

        new_samples = pd.DataFrame(new_samples)
        ans_df = pd.concat([ans_df, new_samples], ignore_index=True)

    ans_dataset = Dataset.from_pandas(ans_df)
    print('Number of illegal samples: ', n_illegal_samples)
    return ans_dataset


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
        sentence = conversation[j]
        if sentence['role'] == 'assistant':
            if j > i:
                assistant_content += '\n'
            assistant_content += conversation[j]['content']
        else:
            break

    example['prompt'] = user_content
    encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
    example['num_tokens'] = len(encoded_response['input_ids'])
    if task_type == 0:
        example['labels'] = len(encoded_response['input_ids'])
    else:
        for i, thresh in enumerate(multi_cls_thresholds):
            if len(encoded_response['input_ids']) < thresh:
                example['labels'] = i
                break
    return example


def tokenize_function(example):
    example = bert_tokenizer(example["prompt"], truncation=False)
    if len(example['input_ids']) >= 512:
        if FLAG_HEAD_TAIL:
            example['input_ids'] = example['input_ids'][: 128] + example['input_ids'][-384: ]
            example['token_type_ids'] = example['token_type_ids'][: 128] + example['token_type_ids'][-384: ]
            example['attention_mask'] = example['attention_mask'][: 128] + example['attention_mask'][-384: ]
        else:
            example['input_ids'] = example['input_ids'][-512: ]
            example['token_type_ids'] = example['token_type_ids'][-512: ]
            example['attention_mask'] = example['attention_mask'][-512: ]
    return example


def replace_model_name_by_idx(example):
    example['model'] = model_name_to_idx[example['model']]
    if task_type == 0:
        # update the model idx with one hot encoding
        # only for task_type == 0 because in other task types, the model idx will be one-hot coded
        # in the recalc_labels_and_one_hot_model_name function
        arr = [0 for _ in range(num_models)]
        arr[example['model']] = 1
        example['model'] = arr
    return example


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
        # sns.histplot(s,
        #          kde=False, 
        #          bins=100, color = 'blue')
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


def preprocess_dataset(dataset):
    dataset = dataset.remove_columns(['openai_moderation', 'redacted', 'language', 'conversation_id', 'turn'])
    new_sentence_column = [''] * len(dataset)
    dataset = dataset.add_column('prompt', new_sentence_column)
    new_label_column = [0] * len(dataset)
    dataset = dataset.add_column('labels', new_label_column)
    if task_type != 0:
        new_length_column = [0] * len(dataset)
        dataset = dataset.add_column('num_tokens', new_length_column)

    # Extract the user prompt(s) and the corresponding response length
    if FLAG_FIRST_ROUND_ONLY:
        dataset = dataset.map(extract_first_round_prompt, remove_columns=['conversation'])
        print('Num samples before filtering: ', len(dataset))
        if task_type == 0:
            dataset = dataset.filter(lambda example: example["labels"] > 1 and example["labels"] < 512)
        else:
            dataset = dataset.filter(lambda example: example["num_tokens"] > 1 and example["num_tokens"] < 512)
        print('Num samples after filtering: ', len(dataset))
    else:
        dataset = exact_multi_round_prompt(dataset)
    if FLAG_VICUNA_DATA_ONLY:
        dataset = dataset.remove_columns(['model'])
    else:
        dataset = dataset.map(replace_model_name_by_idx)
    # Tokenize the user prompt
    # dataset = dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt'])
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_models', action='store_true', default=False)
    parser.add_argument('--multi_round', action='store_true', default=False)
    parser.add_argument('--head_tail', action='store_true', default=False)
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls', default=2)
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=1000)
    parser.add_argument('--model_name', type=str, help='Name of the LLM to predict for', default='vicuna-13b')
    args = parser.parse_args()

    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = args.task_type
    FLAG_VICUNA_DATA_ONLY = not args.all_models
    FLAG_FIRST_ROUND_ONLY = not args.multi_round
    FLAG_HEAD_TAIL = args.head_tail
    # cls_threshold = 328
    if task_type == 1:
        multi_cls_thresholds = [141, 503, 1000000]
    else:
        multi_cls_thresholds = [42, 141, 294, 503, 1000000] if FLAG_FIRST_ROUND_ONLY else [58, 147, 280, 499, 100000]
    dataset_name = 'lmsys/lmsys-chat-1m'
    model_name = 'bert-base-uncased'
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3", legacy=False)  # using vicuna-13b tokenizer for simplicity
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    selected_data_size = 1000 * args.data_size

    model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
                   'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat',
                   'vicuna-7b', 'dolly-v2-12b', 'mpt-30b-chat', 'fastchat-t5-3b', 'chatglm-6b',
                   'claude-1', 'gpt-4', 'vicuna-33b', 'guanaco-33b', 'RWKV-4-Raven-14B',
                   'stablelm-tuned-alpha-7b', 'llama-13b', 'gpt-3.5-turbo', 'llama-2-7b-chat',
                   'claude-2', 'gpt4all-13b-snoozy']
    model_name_to_idx = {model_names[i]: i for i in range(len(model_names))}
    num_models = len(model_names)

    if args.model_name not in model_names:
        print('Model name not found in the list of models:', model_names)
        exit()

    dataset_path = args.model_name.lower()+'_' if FLAG_VICUNA_DATA_ONLY else ''
    dataset_path = dataset_path if task_type == 0 else dataset_path + 'cls_' if task_type == 1 else dataset_path + 'multi_cls_'
    if FLAG_FIRST_ROUND_ONLY:
        dataset_path = 'first_round_data_' + dataset_path
    elif FLAG_HEAD_TAIL:
        dataset_path = 'headtail_' + dataset_path
    else:
        dataset_path = 'tail_' + dataset_path
    dataset_path = 'data/lmsys_' + dataset_path + f'{int(selected_data_size / 1000)}K'

    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.select(range(selected_data_size))
    if FLAG_VICUNA_DATA_ONLY:
        dataset = dataset.filter(lambda example: example["model"] == args.model_name)
    dataset = dataset.shuffle(seed=1)
    dataset = preprocess_dataset(dataset)

    percentiles = [[] for _ in range(num_models)]
    if task_type != 0:
        dataset = calc_percentile(dataset)
    dataset.set_format("torch")

    # print(len(dataset))
    # print(dataset.column_names)
    # print(dataset[0])

    os.makedirs('./data', exist_ok=True)
    dataset.save_to_disk(dataset_path)
    print('Saved dataset to ' + dataset_path)
