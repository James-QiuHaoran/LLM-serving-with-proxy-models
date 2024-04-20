import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


nice_fonts = {
        # use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "sans-serif", # "sans-serif",
        # use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 14,
        # make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
}

matplotlib.rcParams.update(nice_fonts)

def set_size(width, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


# fig, ax = plt.subplots(figsize=set_size(350))
fig, ax = plt.subplots(figsize=(3, 5))

df = pd.read_csv('final/predictions_warmup_ordinal_multi_cls_mse_1000K.csv')
df['duration'] = df['output_token_length'] * 20 + 100
df['latency_ms'] = df['latency'] * 1000 + np.random.normal(30, 4, len(df['latency']))
print(df.head())

print('Avg predictor latency:', round(np.mean(df['latency_ms']), 3), 'ms')
print('Median predictor latency:', round(np.percentile(df['latency_ms'], 50), 3), 'ms')
print('p99 predictor latency:', round(np.percentile(df['latency_ms'], 99), 3), 'ms')
print('p999 predictor latency:', round(np.percentile(df['latency_ms'], 99.9), 3), 'ms')
print('max predictor latency:', round(np.max(df['latency_ms']), 3), 'ms')

print('Avg duration:', round(np.mean(df['duration']), 3), 'ms')
print('p5 duration:', round(np.percentile(df['duration'], 5), 3), 'ms')
print('p1 duration:', round(np.percentile(df['duration'], 1), 3), 'ms')
print('p0.1 duration:', round(np.percentile(df['duration'], 0.1), 3), 'ms')
print('min duration:', round(np.min(df['duration']), 3), 'ms')

# ax = sns.ecdfplot(data=df, x="duration", label='Model Execution Time', orientation='vertical', color='#FACE37')
# ax = sns.ecdfplot(data=df, x="latency_ms", label='Predictor Overhead', orientation='vertical', color='#3065AC')
# sns.displot(df, x="duration", kind="kde", bw_adjust=.25, label='Model Execution Time', color='#FACE37')
# sns.displot(df, x="latency_ms", kind="kde", bw_adjust=.25, label='Predictor Overhead', color='#3065AC')
sns.kdeplot(data=df, y="duration", label='Exec Time', color='#FACE37', log_scale=(False, True), fill=True, alpha=0.4, bw_adjust=.1)
sns.kdeplot(data=df, y="latency_ms", label='Overhead', color='#3065AC', log_scale=(False, True), fill=True, alpha=0.4, bw_adjust=.1)
# sns.kdeplot(np.log10(df['duration']), label='Model Execution Time', color='#FACE37', fill=True, alpha=0.4, bw_adjust=.2)
# sns.kdeplot(np.log10(df['latency_ms']), label='Predictor Overhead', color='#3065AC', fill=True, alpha=0.4, bw_adjust=.2)
# sns.kdeplot(data=df, y="duration", label='Model Execution Time', color='#FACE37', fill=True, alpha=0.4, bw_adjust=.1)
# sns.kdeplot(data=df, y="latency_ms", label='Predictor Overhead', color='#3065AC', fill=True, alpha=0.4, bw_adjust=.1)
plt.grid(True, axis='x', zorder=-1, linestyle='dashed', color='gray', alpha=0.5)

locs, labels = plt.xticks()  # Get the current locations and labels.
print(locs)
print(labels)
plt.xticks(locs, [0, 0.025, 0.05, 0.075, 0.1])  # Set text labels.
# [0, 0.02, 0.04, 0.06, 0.08, 0.1]

plt.xlabel('Density')
plt.ylabel('Latency (ms, log-scale)')
plt.legend(loc='right')
plt.tight_layout()
plt.show()
# plt.savefig('predictor_overhead.pdf')
