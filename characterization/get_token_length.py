import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import time


def get_token_length(sentence):
    encoded_response = vicuna_tokenizer(sentence)
    return len(encoded_response['input_ids'])


if __name__ == '__main__':
    dataset_name = 'lmsys/lmsys-chat-1m'
    vicuna_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")

    dataset = load_dataset(dataset_name, split='train')
    # dataset = dataset.select(range(10000))

    df = pd.DataFrame(columns=['model', 'output_token_length'])
    data = []

    start_time = time.time()

    i = 0
    for example in dataset:
        i += 1
        # print(example['model'])
        conversation = example['conversation']
        for i, sentence in enumerate(conversation):
            # print(sentence['role'], sentence['content'])
            # print(len(sentence['content']))
            token_length = get_token_length(sentence['content'])
            # print(token_length)
            if sentence['role'] == 'assistant':
                data.append({'model': example['model'], 'output_token_length': token_length})
        if i % 1000 == 0:
            print('Processed', i, 'conversations')

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds.")

    print(df.head())

    df.to_csv('token_lengths_all.csv', index=False)
