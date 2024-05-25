# single-round conversations with LLM
# multi-class
python preprocess_dataset.py --task_type 2 --data_size 10
python latency_prediction.py --task_type 2 --data_size 10
python latency_prediction.py --task_type 3 --data_size 10
python latency_prediction.py --task_type 3 --l1_loss --data_size 10

# binary-class
python preprocess_dataset.py --task_type 1 --data_size 10
python latency_prediction.py --task_type 1 --data_size 10

# regression
python preprocess_dataset.py --task_type 0 --data_size 10
python latency_prediction.py --task_type 0 --data_size 10
python latency_prediction.py --task_type 0 --l1_loss --data_size 10

# multi-class
python preprocess_dataset.py --task_type 2 --data_size 10 --all_models
python latency_prediction.py --task_type 2 --data_size 10 --all_models
# regression
python preprocess_dataset.py --task_type 0 --data_size 10 --model_name "gpt-4"
python latency_prediction.py --task_type 0 --data_size 10  --model_name "gpt-4"

# multi-round conversations with LLM
# multi-class
# python preprocess_dataset.py --task_type 2 --multi_round
# python latency_prediction.py --task_type 2 --multi_round
# python latency_prediction.py --task_type 3 --multi_round
# python latency_prediction.py --task_type 3 --multi_round --l1_loss

# binary-class
# python preprocess_dataset.py --task_type 1 --multi_round
# python latency_prediction.py --task_type 1 --multi_round

# regression
# python preprocess_dataset.py --task_type 0 --multi_round
# python latency_prediction.py --task_type 0 --multi_round
# python latency_prediction.py --task_type 0 --multi_round --l1_loss