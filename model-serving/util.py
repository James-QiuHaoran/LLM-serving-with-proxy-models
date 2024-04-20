import os


# SCHEDULEING PARAMS
MODE = 'NO_BATCHING'  # NO_BATCHING, STATIC_BATCHING, DYNAMIC_BATCHING
MAX_BATCH_SIZE = 2
BATCH_WAIT_TIMEOUT = 10
DYNAMIC_BATCHING = False

MODE = os.environ.get('MODE_ENV', 'NOBATCHING')  # Default to NOBATCHING if env var is not set
MAX_BATCH_SIZE = int(os.environ.get('BATCH_SIZE_ENV', 2))  # Default to 2 if env var is not set
DYNAMIC_BATCHING = os.environ.get('DYNAMIC_BATCHING_ENV', 'False').lower() == 'true'  # default False

SJF_STARVATION_WAITING_TIME_THRES = 100  # 50
SJF_STARVATION_WAITING_TIME_COEFFICIENT = 10  # 3

PER_ROUND_EVAL = False
TURN_ID = 0
TURN_ID_COLUMN_NAME = 'turn_id'

PER_ROUND_EVAL = os.environ.get('PER_ROUND_ENV', 'False').lower() == 'true'  # default False
TURN_ID = int(os.environ.get('TURN_ID_ENV', 0))  # Default to 0 if env var is not set

# EVALUATION PARAMS
NUM_EXP_RUNS = 20  # 50
NUM_JOBS = 50

# originally from auto_eval.py
# LLM_DATA_PATH = 'prediction/regression_predictions_vicuna.csv'  # regression
# LLM_DATA_PATH = 'prediction/classification_predictions_vicuna.csv'  # classification (no warmup)
# LLM_DATA_PATH = 'prediction/classification_predictions_vicuna_warmup_1000K.csv'  # classification (warmup)
# LLM_DATA_PATH = 'prediction/classification_predictions_vicuna_warmup_1000K_berttiny.csv'  # classification (warmup, bert-tiny)
# LLM_DATA_PATH = 'prediction/classification_predictions_vicuna_warmup_100K_bertlarge.csv'  # classification (warmup, bert-large)
# LLM_DATA_PATH = 'prediction/classification_multi_predictions_vicuna_1000K.csv'  # multi-class classification
# LLM_DATA_PATH = 'prediction/classification_multi_predictions_vicuna_1000K_berttiny.csv'  # multi-class classification (bert-tiny)
# LLM_DATA_PATH = '../characterization/token_lengths_10000.csv'  # ground truth
# LLM_DATA_PATH = 'prediction/multi-round/predictions_multiround_headtail_warmup_multi_cls_1000K.csv'  # multi-round conversation
# LLM_DATA_PATH = 'prediction/multi-round/predictions_multiround_tail_warmup_multi_cls_1000K.csv'  # multi-round conversation
# CLASSIFIERS = ['cls']

# LLM_DATA_PATH = 'prediction/multi-round/predictions_all.csv'
# CLASSIFIERS = ['head-tail', 'tail-only']

# LLM_DATA_PATH = 'prediction/final/predictions_all.csv'  # multiple predictors new cleaned dataset
LLM_DATA_PATH = 'prediction/final/predictions_multiround_all.csv'  # multiple predictors new cleaned dataset (multi-round)
CLASSIFIERS = ['reg-l1', 'reg-mse', 'cls', 'multi-cls', 'multi-cls-l1', 'multi-cls-mse']

# RESULTS_DIR = 'results/scheduling-algo-barplots/'  # 'results/multi-round/'
# RESULTS_DIR = 'results/cleaned_datasets_no_batching_single_round/'
# RESULTS_DIR = 'results/cleaned_datasets_static_batching_'+str(MAX_BATCH_SIZE)+'_single_round/'
RESULTS_DIR = 'results/cleaned_datasets_dynamic_batch_'+str(MAX_BATCH_SIZE)+'_single_round/'

# RESULTS_DIR = 'results/cleaned_datasets_no_batching_multi_round/'
# RESULTS_DIR = 'results/cleaned_datasets_static_batching_'+str(MAX_BATCH_SIZE)+'_multi_round/'
# RESULTS_DIR = 'results/cleaned_datasets_dynamic_batch_'+str(MAX_BATCH_SIZE)+'_multi_round/'

# if MODE == 'NO_BATCHING':
#     RESULTS_DIR = 'results/cleaned_datasets_no_batching_round_'+str(TURN_ID)+'/'
# elif MODE == 'STATIC_BATCHING':
#     RESULTS_DIR = 'results/cleaned_datasets_static_batching_'+str(MAX_BATCH_SIZE)+'_round_'+str(TURN_ID)+'/'
# elif MODE == 'DYNAMIC_BATCHING':
#     RESULTS_DIR = 'results/cleaned_datasets_dynamic_batch_'+str(MAX_BATCH_SIZE)+'_round_'+str(TURN_ID)+'/'
