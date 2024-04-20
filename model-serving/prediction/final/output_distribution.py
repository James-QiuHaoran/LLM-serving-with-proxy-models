import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('predictions_multiround_all.csv')
print(df.head())

######################
# PDF (density plot) #
######################
print(df[df['model']=='vicuna-13b']['output_token_length'].describe(percentiles=[.25, .5, .75, .8, .9, .95, .97, .99, .999, .9999, .99999]))
sns.distplot(df[(df['model']=='vicuna-13b') & (df['output_token_length']<1024)]['output_token_length'],
             hist=True, kde=False, 
             bins=100, color = 'blue',
             hist_kws={'edgecolor':'black'})
plt.xlabel('Output Token Length')
plt.ylabel('User Requests')
plt.show()

exit()

####################
# CDF and Boxplots #
####################
models = []
for model in df['model'].unique():
    print(model, len(df[df['model']==model]))
    if len(df[df['model']==model]) < 210:
        models.append(model)

df = df[(df['output_token_length'] < 2048) & (df['model'].isin(models))]

# sns.boxplot(data=df, x="model", y="output_token_length")
# plt.xticks(rotation=90)

ax = sns.ecdfplot(data=df, x="output_token_length", hue="model")
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)
sns.set_palette("rocket")
plt.ylabel('CDF')

plt.tight_layout()

plt.show()
