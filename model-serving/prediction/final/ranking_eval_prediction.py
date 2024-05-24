import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score


PER_ROUND_EVAL = False
eval_cases = {
    # 'regression-l1': 'predictions_warmup_reg_l1_1000K.csv',
    # 'regression-mse': 'predictions_warmup_reg_mse_1000K.csv', 
    # 'binary-cls': 'predictions_warmup_cls_1000K.csv',
    # 'multi-cls': 'predictions_warmup_multi_cls_1000K.csv',
    # 'multi-cls-l1': 'predictions_warmup_ordinal_multi_cls_l1_1000K.csv',
    # 'multi-cls-mse': 'predictions_warmup_ordinal_multi_cls_mse_1000K.csv',
    'multi-round-regression-l1': 'predictions_multiround_tail_warmup_reg_l1_1000K.csv',
    'multi-round-regression-mse': 'predictions_multiround_tail_warmup_reg_mse_1000K.csv',
    'multi-round-binary-cls': 'predictions_multiround_tail_warmup_cls_1000K.csv',
    'multi-round-multi-cls': 'predictions_multiround_tail_warmup_multi_cls_1000K.csv',
    'multi-round-multi-cls-l1': 'predictions_multiround_tail_warmup_ordinal_multi_cls_l1_1000K.csv',
    'multi-round-multi-cls-mse': 'predictions_multiround_tail_warmup_ordinal_multi_cls_mse_1000K.csv'
}

# threshold for binary-class classification
binary_cls_thre = 141  # [141, 503, 1000000]

# threshold for multi-class classification
multi_cls_thres_single_round = [58, 147, 280, 499, 1000000]
multi_cls_thres_multi_round = [42, 141, 294, 503, 1000000]

# evaluation for binary classification
def eval_prediction_model_binary_cls(df_all, model_name):
    if model_name == 'all':
        df = df_all
    else:
        df = df_all[df_all['model'] == model_name]

    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        pred = int(row['predicted_length'])
        if pred > 1:
            pred = 1
        label = 1 if int(row['output_token_length']) > binary_cls_thre else 0
        y_true.append(label)
        y_pred.append(pred)

    # print('Binary classification results:')
    # print('Accuracy (single-class):', accuracy_score(y_true, y_pred))
    # print('Precision (single-class):', precision_score(y_true, y_pred))
    # print('Recall (single-class):', recall_score(y_true, y_pred))
    # print('F1 (single-class):', f1_score(y_true, y_pred))

    # evaluate ranking accuracy
    ranking_acc_list = []
    pred_acc_list = []
    for _ in range(300):
        sample_size = 50
        y_true_sample = np.random.choice(y_true, size=sample_size, replace=False)
        y_pred_sample = [y_pred[y_true.index(y)] for y in y_true_sample]

        # calculate prediction accuracy
        pred_acc = sum([1 if y_pred_sample[i] == y_true_sample[i] else 0 for i in range(sample_size)]) / sample_size
        pred_acc_list.append(pred_acc)

        y_true_sorted_indices = sorted(range(len(y_true_sample)), key=lambda i: y_true_sample[i])
        y_pred_sorted_indices = sorted(range(len(y_pred_sample)), key=lambda i: y_pred_sample[i])

        # weighted ranking accuracy
        total_weight = sum(y_true_sample)
        ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j] * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
                                else -1 * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                for i in range(len(y_true_sample))
                                for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        # ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j]
        #                         if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
        #                         else y_true_sorted_indices[i] < y_true_sorted_indices[j]
        #                         for i in range(len(y_true_sample))
        #                         for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        ranking_acc_list.append(ranking_accuracy)

    print(f"Ranking accuracy (based on per-{sample_size} random samples): {np.mean(ranking_acc_list):.4f}")
    print(f"Prediction accuracy (based on per-{sample_size} random samples): {np.mean(pred_acc_list):.4f}")
    print()


# evaluation for multi-class classification
def eval_prediction_model_multi_cls(df_all, model_name, multi_round=False, from_regression=False):
    if model_name == 'all':
        df = df_all
    else:
        df = df_all[df_all['model'] == model_name]

    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        multi_cls_thres = multi_cls_thres_multi_round if multi_round else multi_cls_thres_single_round
        if from_regression:
            # from regression
            for i, thre in enumerate(multi_cls_thres):
                if int(row['predicted_length']) <= thre:
                    pred = i
                    break
                else:
                    pred = i
        else:
            # from ordinal multi-cls and standard multi-cls
            pred = int(row['predicted_length'])
        for i, thre in enumerate(multi_cls_thres):
            if int(row['output_token_length']) <= thre:
                label = i
                break
        y_true.append(label)
        y_pred.append(pred)

    # print('Multi-class classification results:')
    # print('Accuracy (multi-class):', accuracy_score(y_true, y_pred))
    # f1_avg = f1_score(y_true, y_pred, average='macro')
    # print('F1 (multi-class) (macro):', f1_avg)
    # print('F1 (multi-class) (micro):', f1_score(y_true, y_pred, average='micro'))
    # print('F1 (multi-class) (weighted):', f1_score(y_true, y_pred, average='weighted'))
    # print('F1 (multi-class) (per-class):', f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4]))
    # # print the percentage of each classes in all the labels
    # print('Percentage of each class in all the labels:')
    # for i in range(5):
    #     print('Class', i, ':', len([j for j in y_true if j == i]) / len(y_true))
    # print('Precision (multi-class) (weighted):', precision_score(y_true, y_pred, average='weighted', zero_division=np.nan))
    # print('Recall (multi-class) (weighted):', recall_score(y_true, y_pred, average='weighted'))

    # evaluate ranking accuracy
    ranking_acc_list = []
    pred_acc_list = []
    for _ in range(300):
        sample_size = 50
        y_true_sample = np.random.choice(y_true, size=sample_size, replace=False)
        y_pred_sample = [y_pred[y_true.index(y)] for y in y_true_sample]

        # calculate prediction accuracy
        pred_acc = sum([1 if y_pred_sample[i] == y_true_sample[i] else 0 for i in range(sample_size)]) / sample_size
        pred_acc_list.append(pred_acc)

        y_true_sorted_indices = sorted(range(len(y_true_sample)), key=lambda i: y_true_sample[i])
        y_pred_sorted_indices = sorted(range(len(y_pred_sample)), key=lambda i: y_pred_sample[i])

        # weighted ranking accuracy
        total_weight = sum(y_true_sample)
        ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j] * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
                                else -1 * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                for i in range(len(y_true_sample))
                                for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        # ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j]
        #                         if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
        #                         else y_true_sorted_indices[i] < y_true_sorted_indices[j]
        #                         for i in range(len(y_true_sample))
        #                         for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        ranking_acc_list.append(ranking_accuracy)

    print(f"[Multi-class] Ranking accuracy (based on per-{sample_size} random samples): {np.mean(ranking_acc_list):.4f}")
    print(f"[Multi-class] Prediction accuracy (based on per-{sample_size} random samples): {np.mean(pred_acc_list):.4f}")
    print()


# evaluation for regression
def eval_prediction_model_regression(df_all, model_name):
    if model_name == 'all':
        df = df_all
    else:
        df = df_all[df_all['model'] == model_name]

    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        y_true.append(float(row['output_token_length']))
        y_pred.append(float(row['predicted_length']))

    # evaluate ranking accuracy
    ranking_acc_list = []
    pred_acc_list = []
    for _ in range(300):
        sample_size = 50
        y_true_sample = np.random.choice(y_true, size=sample_size, replace=False)
        y_pred_sample = [y_pred[y_true.index(y)] for y in y_true_sample]

        # calculate prediction accuracy
        pred_acc = sum([1 if y_pred_sample[i] == y_true_sample[i] else 0 for i in range(sample_size)]) / sample_size
        pred_acc_list.append(pred_acc)

        y_true_sorted_indices = sorted(range(len(y_true_sample)), key=lambda i: y_true_sample[i])
        y_pred_sorted_indices = sorted(range(len(y_pred_sample)), key=lambda i: y_pred_sample[i])

        # ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j]
        #                         if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
        #                         else y_true_sorted_indices[i] < y_true_sorted_indices[j]
        #                         for i in range(len(y_true_sample))
        #                         for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        # weighted ranking accuracy
        total_weight = sum(y_true_sample)
        ranking_accuracy = sum([y_true_sorted_indices[i] >= y_true_sorted_indices[j] * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                if y_pred_sorted_indices[i] >= y_pred_sorted_indices[j]
                                else -1 * y_true_sample[y_true_sorted_indices[i]] / total_weight
                                for i in range(len(y_true_sample))
                                for j in range(i)]) / (sample_size * (sample_size - 1) / 2)
        ranking_acc_list.append(ranking_accuracy)

    print(f"[Regression] Ranking accuracy (based on per-{sample_size} random samples): {np.mean(ranking_acc_list):.4f}")
    print(f"[Regression] Prediction accuracy (based on per-{sample_size} random samples): {np.mean(pred_acc_list):.4f}")
    print()


### Single-round Binary-class Classification Evaluation ###
if 'binary-cls' in eval_cases:
    df = pd.read_csv(eval_cases['binary-cls'])
    # print(df.head())
    print('>>> Single-round Binary-class Classification Evaluation <<<')
    eval_prediction_model_binary_cls(df, 'vicuna-13b')


### Single-round Multi-class Classification Evaluation ###
if 'multi-cls' in eval_cases:
    df = pd.read_csv(eval_cases['multi-cls'])
    # print(df.head())
    print('>>> Single-round Multi-class Classification Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=False)


### Single-round Multi-class Classification (MSE) Evaluation ###
if 'multi-cls-mse' in eval_cases:
    df = pd.read_csv(eval_cases['multi-cls-mse'])
    # print(df.head())
    print('>>> Single-round Multi-class Classification (MSE) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=False)


### Single-round Multi-class Classification (L1) Evaluation ###
if 'multi-cls-l1' in eval_cases:
    df = pd.read_csv(eval_cases['multi-cls-l1'])
    # print(df.head())
    print('>>> Single-round Multi-class Classification (L1) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=False)


### Single-round Regression (MSE) Evaluation ###
if 'regression-mse' in eval_cases:
    df = pd.read_csv(eval_cases['regression-mse'])
    # print(df.head())
    print('>>> Single-round Regression (MSE) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=False, from_regression=True)
    eval_prediction_model_regression(df, 'vicuna-13b')


### Single-round Regression (L1) Evaluation ###
if 'regression-l1' in eval_cases:
    df = pd.read_csv(eval_cases['regression-l1'])
    # print(df.head())
    print('>>> Single-round Regression (L1) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=False, from_regression=True)
    eval_prediction_model_regression(df, 'vicuna-13b')


### Multi-round Binary-class Classification Evaluation ###
if 'multi-round-binary-cls' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-binary-cls'])
    # print(df.head())
    print('>>> Multi-round Binary-class Classification Evaluation <<<')
    eval_prediction_model_binary_cls(df, 'vicuna-13b')


### Multi-round Multi-class Classification Evaluation ###
if 'multi-round-multi-cls' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-multi-cls'])
    # print(df.head())
    print('>>> Multi-round Multi-class Classification Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=True)


### Multi-round Multi-class Classification (MSE) Evaluation ###
if 'multi-round-multi-cls-mse' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-multi-cls-mse'])
    # print(df.head())
    print('>>> Multi-round Multi-class Classification (MSE) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=True)


### Multi-round Multi-class Classification (L1) Evaluation ###
if 'multi-round-multi-cls-l1' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-multi-cls-l1'])
    # print(df.head())
    print('>>> Multi-round Multi-class Classification (L1) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=True)


### Multi-round Regression (MSE) Evaluation ###
if 'multi-round-regression-mse' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-regression-mse'])
    # print(df.head())
    print('>>> Multi-round Regression (MSE) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=True, from_regression=True)


### Multi-round Regression (L1) Evaluation ###
if 'multi-round-regression-l1' in eval_cases:
    df = pd.read_csv(eval_cases['multi-round-regression-l1'])
    # print(df.head())
    print('>>> Multi-round Regression (L1) Evaluation <<<')
    eval_prediction_model_multi_cls(df, 'vicuna-13b', multi_round=True, from_regression=True)

    # per-round evaluation
    if PER_ROUND_EVAL:
        print('>>> Per-round Regression (L1) Evaluation <<<')
        for i in range(5):
            df_round = df[df['turn_id'] == i]
            print('>>> Round', i, '<<<')
            eval_prediction_model_multi_cls(df_round, 'vicuna-13b', multi_round=True, from_regression=True)
