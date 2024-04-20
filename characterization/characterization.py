import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('token_lengths_all.csv')
print(df.head())

# get some quantifications
for model in df['model'].unique().tolist():
    print(model, len(df[df['model']==model]))
    print('p50', df[df['model']==model]['output_token_length'].quantile(0.5))
    print('p90', df[df['model']==model]['output_token_length'].quantile(0.9))
    print('p95', df[df['model']==model]['output_token_length'].quantile(0.95))
    print('p95 / p50 =', df[df['model']==model]['output_token_length'].quantile(0.95) / df[df['model']==model]['output_token_length'].quantile(0.5))

df = df[(df['output_token_length'] < 2048)]
df = df.replace('stablelm-tuned-alpha-7b', 'stablelm-alpha-7b')

fig, ax = plt.subplots(figsize=(6.4, 3.6))
sns.boxplot(data=df, x="model", y="output_token_length", fliersize=3, showfliers=False, whis=1.5)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Output Token Length')
# sns.color_palette("Spectral", as_cmap=True)
plt.grid(True, axis='y', zorder=-1, linestyle='dashed', color='gray', alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('token_length_boxplots_all.pdf')
