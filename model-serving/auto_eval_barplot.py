import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from util import *


nice_fonts = {
        # use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "sans-serif", # "sans-serif",
        # use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 14,
        # make the legend/label fonts a little smaller
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
}

matplotlib.rcParams.update(nice_fonts)

SAVE_FIGURES = True
LEANED_SAVED_FIGURES = True
CACHING = False
REMOVING_L1 = False

# avg or 99th percentile
AVG = True
PERCENTILE = 99

exps = {
    1: RESULTS_DIR + 'results-poisson-varying-rates-vicuna-'+str(NUM_JOBS)+'jobs.csv',
    # 2: RESULTS_DIR + 'results-poisson-varying-num-jobs-vicuna-rate-05.csv',
    # 3: RESULTS_DIR + 'results-gamma-varying-rates-vicuna-'+str(NUM_JOBS)+'jobs-cv-2.csv',
    4: RESULTS_DIR + 'results-gamma-varying-cv-vicuna-'+str(NUM_JOBS)+'jobs-rate-003.csv',
    # 5: RESULTS_DIR + 'results-gamma-varying-num-jobs-vicuna-rate-003-cv-2.csv',
    # 6: RESULTS_DIR + 'results-poisson-varying-models-20jobs-rate-1.csv',
    # 7: RESULTS_DIR + 'results-gamma-varying-model-20jobs-rate-005-cv-2.csv'
    8: RESULTS_DIR + 'results-azure-varying-scales-vicuna-'+str(NUM_JOBS)+'jobs.csv',
}
fig_size_w = 12
fig_size_h = 4
num_cols = 3
x_ticks_reduction = 2  # used to reduce the number of boxes in boxplots
width = 0.1  # the width of the bars
cache = 0.23


def key_figure(distribution, key, x_key):
    if distribution == 'poisson' and 'jct' in key and x_key == 'arrival_rate':
        return True
    elif distribution == 'poisson' and key == 'throughput' and x_key == 'arrival_rate':
        return True
    elif distribution == 'gamma' and 'jct' in key and x_key == 'cv':
        return True
    elif distribution == 'gamma' and key == 'throughput' and x_key == 'cv':
        return True
    elif distribution == 'azure' and 'jct' in key and x_key == 'scale':
        return True
    elif distribution == 'azure' and key == 'throughput' and x_key == 'scale':
        return True
    else:
        return False


def draw_barplots(df, x_key, x_label, distribution='poisson', avg=AVG, percentile=PERCENTILE):
    keys = ['jct', 'waiting_time', 'throughput']
    ylabels = ['JCT', 'Waiting Time', 'Throughput']
    colors = ['#3065AC', '#2B7BBA', '#009DCC', '#15B1BA', '#A0BC71', '#FACE37', '#F2EA3B', 'indianred']
    # bar plots - JCT, waiting time, throughput
    for key in keys:
        x_items = df[x_key].unique().tolist()
        average_dict = {}
        std_dict = {}
        algorithms = df['algorithm'].unique().tolist()
        for algorithm in algorithms:
            average_dict[algorithm] = []
            std_dict[algorithm] = []
            for x_item in x_items:
                if not avg and key == 'jct':
                    # convert key from jct to p99_jct
                    key = 'p' + str(int(percentile)) + '_jct'
                    average_dict[algorithm].append(np.mean(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key]))
                    # average_dict[algorithm].append(np.percentile(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key], percentile))
                else:
                    average_dict[algorithm].append(np.mean(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key]))
                std_dict[algorithm].append(np.std(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key]) / 3)
        x = np.arange(len(x_items))  # the label locations
        multiplier = 0

        _, ax = plt.subplots(figsize=(fig_size_w, fig_size_h))
        # ax.yaxis.grid(True)
        for algorithm, measurements in average_dict.items():
            if algorithm == 'with-cache':
                plt.bar(x + multiplier * width, measurements, width, label=algorithm, yerr=std_dict[algorithm], color=colors[multiplier], hatch='xxx', edgecolor='black')
            else:
                plt.bar(x + multiplier * width, measurements, width, label=algorithm, yerr=std_dict[algorithm], color=colors[multiplier])
            multiplier += 1

        plt.xlabel(x_label)
        if not avg and 'jct' in key:
            plt.ylabel('P'+str(int(percentile)) + ' ' + ylabels[keys.index('jct')])
        else:
            plt.ylabel('Avg ' + ylabels[keys.index(key)])
        plt.legend(ncols=num_cols)
        plt.grid(True, axis='y', zorder=-1, linestyle='dashed', color='gray', alpha=0.5)
        plt.xticks(x + width * (multiplier - 1) / 2, x_items)
        plt.tight_layout()
        if SAVE_FIGURES:
            if LEANED_SAVED_FIGURES and not key_figure(distribution, key, x_key):
                pass
            else:
                if key == 'throughput':
                    plt.savefig(RESULTS_DIR + 'barplot-' + distribution + '-' + key + '-vs-' + x_key + '.pdf')
                elif avg:
                    plt.savefig(RESULTS_DIR + 'barplot-' + distribution + '-' + key + '-vs-' + x_key + '-avg.pdf')
                else:
                    plt.savefig(RESULTS_DIR + 'barplot-' + distribution + '-' + key + '-vs-' + x_key + '-p' + str(int(percentile)) + '.pdf')
        else:
            plt.show()


def report_stats(df, x_key, distribution='poisson', avg=AVG, percentile=PERCENTILE):
    print('Stats for distribution', distribution, 'with x_key', x_key)
    keys = ['jct', 'throughput']
    for key in keys:
        x_items = df[x_key].unique().tolist()
        average_dict = {}
        algorithms = df['algorithm'].unique().tolist()
        for algorithm in algorithms:
            average_dict[algorithm] = []
            for x_item in x_items:
                if not avg and key == 'jct':
                    # convert key from jct to p99_jct
                    key = 'p' + str(int(percentile)) + '_jct'
                    average_dict[algorithm].append(np.mean(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key]))
                    # average_dict[algorithm].append(np.percentile(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key], percentile))
                else:
                    average_dict[algorithm].append(np.mean(df[(df[x_key] == x_item) & (df['algorithm'] == algorithm)][key]))
        print('Key:', key)
        avg_fcfs = np.mean(average_dict['fcfs'])
        for algorithm in algorithms:
            if algorithm == 'fcfs':
                continue
            else:
                print('Improvement of', algorithm, 'on top of FCFS:')
                if 'jct' in key:
                    avg_jct = np.mean(average_dict[algorithm])
                    print('>>>', round((avg_fcfs - avg_jct) / avg_fcfs * 100, 2), '%')
                elif key == 'throughput':
                    avg_throughput = np.mean(average_dict[algorithm])
                    print('>>>', round(avg_throughput / avg_fcfs, 2))

if 1 in exps:
    # EXP #1 - experiments on poisson distribution with varying arrival rates
    print('>>> EXP #1 - experiments on poisson distribution with varying arrival rates')
    df = pd.read_csv(exps[1])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['arrival_rate'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['arrival_rate'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'arrival_rate', 'Arrival Rate', distribution='poisson')
    report_stats(df, 'arrival_rate', distribution='poisson')

if 2 in exps:
    # EXP #2 - experiments on poisson distribution with varying number of jobs
    print('>>> EXP #2 - experiments on poisson distribution with varying number of jobs')
    df = pd.read_csv(exps[2])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['num_jobs'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['num_jobs'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'num_jobs', 'Num of Jobs', distribution='poisson')

if 3 in exps:
    # EXP #3 - experiments on gamma distribution with varying arrival rates
    print('>>> EXP #3 - experiments on gamma distribution with varying arrival rates')
    df = pd.read_csv(exps[3])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['arrival_rate'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['arrival_rate'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'arrival_rate', 'Arrival Rate', distribution='gamma')

if 4 in exps:
    # EXP #4 - experiments on gamma distribution with varying CVs
    print('>>> EXP #4 - experiments on gamma distribution with varying CVs')
    df = pd.read_csv(exps[4])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['cv'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['cv'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'cv', 'Coefficient of Variance', distribution='gamma')
    report_stats(df, 'cv', distribution='gamma')

if 5 in exps:
    # EXP #5 - experiments on gamma distribution with varying number of jobs
    print('>>> EXP #5 - experiments on gamma distribution with varying number of jobs')
    df = pd.read_csv(exps[5])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['num_jobs'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['num_jobs'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'num_jobs', 'Num of Jobs', distribution='gamma')

if 8 in exps:
    # EXP #8 - experiments on Azure traces with varying scales
    print('>>> EXP #8 - experiments on Azure traces with varying scales')
    df = pd.read_csv(exps[8])
    # print(df.head())

    # reduced boxes in boxplot
    all_boxes = list(df['scale'].unique())
    boxes_to_plot = [elem for i, elem in enumerate(all_boxes) if i % x_ticks_reduction == 0]
    print(all_boxes, '->', boxes_to_plot)
    df = df[df['scale'].isin(boxes_to_plot)]

    # remove l1-based algo
    if REMOVING_L1:
        df = df[~df['algorithm'].str.contains('l1')]

    draw_barplots(df, 'scale', 'Intensity Scale', distribution='azure')
    report_stats(df, 'scale', distribution='azure')