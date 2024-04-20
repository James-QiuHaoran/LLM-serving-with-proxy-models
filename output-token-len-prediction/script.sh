# python preprocess_dataset.py --task_type 2
# python latency_prediction.py --task_type 2
# python preprocess_dataset.py --task_type 1
# python latency_prediction.py --task_type 1
# python preprocess_dataset.py --task_type 0
# python latency_prediction.py --task_type 0
# python latency_prediction.py --task_type 0 --l1_loss
# python latency_prediction.py --task_type 3
# python latency_prediction.py --task_type 3 --l1_loss

# multi-class
# python preprocess_dataset.py --task_type 2 --multi_round
# python latency_prediction.py --task_type 2 --multi_round
# python latency_prediction.py --task_type 3 --multi_round
# python latency_prediction.py --task_type 3 --multi_round --l1_loss

# binary-class
# python preprocess_dataset.py --task_type 1 --multi_round
# python latency_prediction.py --task_type 1 --multi_round

# regression
python preprocess_dataset.py --task_type 0 --multi_round
python latency_prediction.py --task_type 0 --multi_round
python latency_prediction.py --task_type 0 --multi_round --l1_loss