import pandas as pd
from tqdm import tqdm


df_list = []
MULTI_ROUND = True

if not MULTI_ROUND:
    # single-round LLM conversation
    df_list.append(pd.read_csv('predictions_warmup_reg_l1_1000K.csv'))
    df_list.append(pd.read_csv('predictions_warmup_reg_mse_1000K.csv'))
    df_list.append(pd.read_csv('predictions_warmup_cls_1000K.csv'))
    df_list.append(pd.read_csv('predictions_warmup_multi_cls_1000K.csv'))
    df_list.append(pd.read_csv('predictions_warmup_ordinal_multi_cls_l1_1000K.csv'))
    df_list.append(pd.read_csv('predictions_warmup_ordinal_multi_cls_mse_1000K.csv'))

    classifiers = ['reg-l1', 'reg-mse', 'cls', 'multi-cls', 'multi-cls-l1', 'multi-cls-mse']
else:
    # multi-round LLM conversation
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_reg_l1_1000K.csv'))
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_reg_mse_1000K.csv'))
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_cls_1000K.csv'))
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_multi_cls_1000K.csv'))
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_ordinal_multi_cls_l1_1000K.csv'))
    df_list.append(pd.read_csv('predictions_multiround_tail_warmup_ordinal_multi_cls_mse_1000K.csv'))

    classifiers = ['reg-l1', 'reg-mse', 'cls', 'multi-cls', 'multi-cls-l1', 'multi-cls-mse']

columns = ['output_token_length', 'model']
if MULTI_ROUND:
    columns.append('turn_id')

for classifier in classifiers:
    columns.append(classifier + '_prediction')
    columns.append(classifier + '_overhead')
final_df = pd.DataFrame(columns=columns)

print('Before merging:', len(df_list[0]))

for index, row in tqdm(df_list[0].iterrows()):
    row_item = row.to_dict()
    row_all = {'output_token_length': row_item['output_token_length'],
               'model': row_item['model']}
    if MULTI_ROUND:
        row_all['turn_id'] = row_item['turn_id']
    row_all[classifiers[0] + '_prediction'] = row_item['predicted_length']
    row_all[classifiers[0] + '_overhead'] = row_item['latency']
    for classifier_i in range(1, len(classifiers)):
        classifier = classifiers[classifier_i]
        row_item = df_list[classifier_i].iloc[index].to_dict()
        if row_item['output_token_length'] != row_all['output_token_length']:
            print('Error! Length of the output token sequence not consistent!')
            print(classifier, row_item, '\n', row_all)
            exit()
        if MULTI_ROUND and row_item['turn_id'] != row_all['turn_id']:
            print('Error! turn_id not consistent!')
            print(classifier, row_item, '\n', row_all)
            exit()
        row_all[classifier + '_prediction'] = row_item['predicted_length']
        row_all[classifier + '_overhead'] = row_item['latency']
    final_df = pd.concat([final_df, pd.DataFrame([row_all])], ignore_index=True)


print('After merging:', len(final_df))
print(final_df.head())
if MULTI_ROUND:
    final_df.to_csv('predictions_multiround_all.csv')
else:
    final_df.to_csv('predictions_all.csv')
