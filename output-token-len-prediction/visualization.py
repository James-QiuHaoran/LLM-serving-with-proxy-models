import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def get_actual_label(x, num_classes):
    if num_classes == 2:
        return 0 if x <= 500 else 1
    else:
        if x <= 200:
            return 0
        elif x <= 500:
            return 1
        elif x <= 800:
            return 2
        else:
            return 3


def get_actual_label_new_div(x, num_classes):
    if num_classes == 2:
        return 0 if x <= 183 else 1
    else:
        if x <= 73:
            return 0
        elif x <= 183:
            return 1
        elif x <= 337:
            return 2
        elif x <= 810:
            return 3
        else:
            return 4


def gen_prediction_heatmap():
    num_classes = 2
    df = pd.read_csv('results/predictions_warmup_cls_1000K.csv')

    actual_len = df['actual_length'].to_numpy()
    prediction = df['predicted_label'].to_numpy()

    count = [[0 for _ in range(num_classes)] for __ in range(num_classes)]
    for i in range(len(actual_len)):
        actual_len[i] = get_actual_label(actual_len[i], num_classes)
        count[actual_len[i]][prediction[i]] += 1

    Index= [f'actual_{i}' for i in range(num_classes)]
    Col = [f'predicted_{i}' for i in range(num_classes)]

    count = pd.DataFrame(np.array(count), index=Index, columns=Col)
    sns.heatmap(count, annot=True)
    plt.title('Actual labels vs. predicted labels')
    plt.savefig('heatmap_binary.png')


def calc_accuracy_vs_round():
    num_classes = 5
    df = pd.read_csv('results/predictions_multiround_headtail_warmup_multi_cls_1000K.csv')

    actual_label = df['actual_length'].tolist()
    predicted_label = df['predicted_label'].tolist()
    for sample_i in range(len(actual_label)):
        actual_label[sample_i] = get_actual_label_new_div(actual_label[sample_i], num_classes)
    print(f'Overall: ')
    accuracy = accuracy_score(actual_label, predicted_label)
    f1 = f1_score(actual_label, predicted_label, average='weighted')
    print(f'Accuracy: {accuracy}, F1 score: {f1}')

    for round_i in range(5):
        df_i = df[df['turn_id'] == round_i]
        actual_label = df_i['actual_length'].tolist()
        predicted_label = df_i['predicted_label'].tolist()
        for sample_i in range(len(actual_label)):
            actual_label[sample_i] = get_actual_label_new_div(actual_label[sample_i], num_classes)
        print(f'Round {round_i}: ')
        accuracy = accuracy_score(actual_label, predicted_label)
        f1 = f1_score(actual_label, predicted_label, average='weighted')
        print(f'Accuracy: {accuracy}, F1 score: {f1}')


if __name__ == '__main__':
    calc_accuracy_vs_round()