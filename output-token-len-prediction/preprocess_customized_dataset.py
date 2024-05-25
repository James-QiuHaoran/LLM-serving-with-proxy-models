from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os


def tokenize_function(example):
    example = bert_tokenizer(example["prompt_content"], truncation=False)
    if len(example['input_ids']) >= 512:
        example['input_ids'] = example['input_ids'][-512: ]
        example['token_type_ids'] = example['token_type_ids'][-512: ]
        example['attention_mask'] = example['attention_mask'][-512: ]
    return example


# def replace_model_name_by_idx(example):
#     example['model'] = model_name_to_idx[example['model']]
#     if task_type == 0:
#         # update the model idx with one hot encoding
#         # only for task_type == 0 because in other task types, the model idx will be one-hot coded
#         # in the recalc_labels_and_one_hot_model_name function
#         arr = [0 for _ in range(num_models)]
#         arr[example['model']] = 1
#         example['model'] = arr
#     return example


# def recalc_labels_and_one_hot_model_name(example):
#     for i, thresh in enumerate(percentiles[example['model']]):
#         if example['num_tokens'] < thresh:
#             example['labels'] = i
#             break
#     arr = [0 for _ in range(num_models)]
#     arr[example['model']] = 1
#     example['model'] = arr
#     return example


def calc_percentile(dataset):
    if FLAG_SINGLE_MODEL:
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
        # raise not implemented error for multi-model datasets
        raise NotImplementedError('The support for multi-model datasets has not been implemented yet!')
        # output_token_lengths = [[] for _ in range(num_models)]
        # for sample in dataset:
        #     output_token_lengths[sample['model']].append(sample['num_tokens'])
        # for model_id in range(num_models):
        #     s = pd.Series(output_token_lengths[model_id])
        #     desc = s.describe(percentiles=[.25, .5, .75, .99])
        #     percentiles[model_id].extend([desc['25%'], desc['50%'], desc['75%'], desc['99%'], 1000000])
        # # print(percentiles)
        # dataset = dataset.map(recalc_labels_and_one_hot_model_name)
    return dataset


def preprocess_dataset(dataset):
    """
    Preprocess a given dataset to convert to huggingface dataset format.
    """
    dataset = Dataset.from_pandas(dataset)

    # check if the dataset contains columns: model_name, prompt_id, prompt_content, response_length
    if not all(col in dataset.column_names for col in ['model_name', 'prompt_id', 'prompt_content', 'response_length']):
        print('Dataset does not contain required columns: model_name, prompt_id, prompt_content, response_length')
        exit()

    # remove all other columns
    dataset = dataset.select_columns(['model_name', 'prompt_id', 'prompt_content', 'response_length'])

    if task_type == 0:
        # rename the column name from response_length to num_tokens
        dataset = dataset.rename_column('response_length', 'num_tokens')
    else:
        # rename the column name from response_length to labels
        dataset = dataset.rename_column('response_length', 'labels')

    # remove or encode model name column
    if FLAG_SINGLE_MODEL:
        dataset = dataset.remove_columns(['model_name'])
    else:
        # dataset = dataset.map(replace_model_name_by_idx)  # replace model name with model idx with one-hot encoding
        # raise not implemented error for multi-model datasets
        raise NotImplementedError('The support for multi-model datasets has not been implemented yet!')

    # tokenize the user prompt for proxy model processing
    # dataset = dataset.map(tokenize_function, batched=True, remove_columns=['prompt_content'])
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=['prompt_content'])
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_model', action='store_true', default=True)
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset file', default='sampled_prompts_with_labels.csv')
    parser.add_argument('--task_type', type=int, help='0 for regression, 1 for binary cls, 2 for multi-cls', default=2)
    parser.add_argument('--data_size', type=int, help='Size of the dataset to use (in thousands)', default=1000)
    args = parser.parse_args()

    FLAG_SINGLE_MODEL = args.single_model

    # 0: regression; 1: binary classification; 2: multi-class classification;
    task_type = args.task_type
    if task_type == 1:
        multi_cls_thresholds = [141, 503, 1000000]  # threshold will be adjusted automatically
    else:
        multi_cls_thresholds = [42, 141, 294, 503, 1000000]  # threshold will be adjusted automatically

    selected_data_size = 1000 * args.data_size
    dataset_path = args.dataset_path
    # read the first selected_data_size rows from the dataset path into dataframe
    dataset = pd.read_csv(dataset_path, nrows=selected_data_size)

    # proxy-model
    proxy_model_name = 'bert-base-uncased'
    bert_tokenizer = AutoTokenizer.from_pretrained(proxy_model_name)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # for multi-model datasets
    # model_names = ['vicuna-13b', 'wizardlm-13b', 'palm-2', 'llama-2-13b-chat', 'koala-13b',
    #                'claude-instant-1', 'oasst-pythia-12b', 'alpaca-13b', 'mpt-7b-chat']
    # model_name_to_idx = {model_names[i]: i for i in range(len(model_names))}
    # num_models = len(model_names)
    # percentiles = [[] for _ in range(num_models)]

    dataset = preprocess_dataset(dataset)

    if task_type != 0:
        dataset = calc_percentile(dataset)
    dataset.set_format("torch")

    dataset_path = "" if task_type == 0 else dataset_path + 'cls_' if task_type == 1 else dataset_path + 'multi_cls_'
    dataset_path = 'data/customized_' + dataset_path + f'{int(selected_data_size / 1000)}K'

    os.makedirs('./data', exist_ok=True)
    dataset.save_to_disk(dataset_path)
    print('Saved dataset to ' + dataset_path)
