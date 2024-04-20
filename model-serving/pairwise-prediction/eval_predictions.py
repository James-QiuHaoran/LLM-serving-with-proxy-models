import pandas as pd
import numpy as np


file_name = 'pairwise_warmup_1000K_grouped.csv'
# file_name = 'pairwise_warmup_1000K.csv'

df = pd.read_csv(file_name)
print(df.head())

# print('Total number of groups/batches:', len(df[df['model']=='vicuna-13b']['group_id'].unique()))
# job_ids_2 = df[df['group_id'] == 0]['sample_id_2'].unique().tolist()
# job_ids_1 = df[df['group_id'] == 0]['sample_id_1'].unique().tolist()
# job_ids_1.extend(job_ids_2)
# all_jobs = set(job_ids_1)
# print(sorted(all_jobs))
# print('Total number of jobs:', len(all_jobs))

TT, TF, FT, FF = 0, 0, 0, 0
for idx, row in df.iterrows():
    pred = int(row['predicted_label'])
    label = 1 if int(row['actual_length1']) > int(row['actual_length2']) else 0
    if label == 1 and pred == 1:
        TT +=1
    elif label == 1 and pred == 0:
        TF += 1
    elif label == 0 and pred == 1:
        FT += 1
    elif label == 0 and pred == 0:
        FF += 1

print('Pairwise binary classification results:')
print('Accuracy:', round((TT + FF) / (TT + TF + FT + FF), 2))
precision = TT / (TT + FT)
print('Precision:', round(TT / (TT + FT), 2))
recall = TT / (TT + TF)
print('Recall:', round(TT / (TT + TF), 2))
print('F1:', round(2 * (precision * recall) / (precision + recall), 5))