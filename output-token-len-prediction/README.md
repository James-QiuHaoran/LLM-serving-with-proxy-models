# llm-latency-prediction

To run:
```
accelerate launch latency_prediction.py
```

By default it's for the LMSYS-Chat-1M dataset.

## Customized Datasets

If you have collected your own dataset for LLM input-output history, you can use the following steps to customize the predictor training/inference for your own dataset.

1. Make sure you have the necessary columns: `model_name`, `prompt_id`, `prompt_content`, `response_length`
2. Prepare the dataset that the predictor can process: `python preprocess_customized_dataset.py --single_model --task_type 0 --data_size 1 --dataset_path <path_to_your_dataset>`
3. Train the predictor: `python latency_prediction.py --task_type 0 --data_size 1 --customized --dataset_path <path_to_your_processed_dataset>`

For example:

```
python preprocess_customized_dataset.py --single_model --task_type 0 --data_size 1 --dataset_path test_customized_dataset.csv
python latency_prediction.py --task_type 0 --data_size 1 --customized --dataset_path data/customized_1K
```