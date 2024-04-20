import numpy as np
import pandas as pd
import os

from eval import run_exp
from util import *


def init_map(algorithms):
    map_to_return = {}
    for algo in algorithms:
        if algo != 'sjfp' or len(CLASSIFIERS) == 1:
            map_to_return[algo] = []
        else:
            for classifier in CLASSIFIERS:
                map_to_return[algo+'-'+classifier] = []
    return map_to_return


def main():
    algorithms = ['fcfs', 'sjf', 'sjfp']
    # arrival_rates_poisson = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    # arrival_rates_poisson = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    arrival_rates_poisson = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_jobs_list = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    arrival_rates_gamma = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
    cv_arrival_rates_gamma = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

    # experiments and result CSV file names
    exps = {
        1: RESULTS_DIR + 'results-poisson-varying-rates-vicuna-'+str(NUM_JOBS)+'jobs.csv',
        # 2: RESULTS_DIR + 'results-poisson-varying-num-jobs-vicuna-rate-05.csv',
        # 3: RESULTS_DIR + 'results-gamma-varying-rates-vicuna-'+str(NUM_JOBS)+'jobs-cv-2.csv',
        4: RESULTS_DIR + 'results-gamma-varying-cv-vicuna-'+str(NUM_JOBS)+'jobs-rate-003.csv',
        # 5: RESULTS_DIR + 'results-gamma-varying-num-jobs-vicuna-rate-003-cv-2.csv',
        # 6: RESULTS_DIR + 'results-poisson-varying-models-20jobs-rate-1.csv',
        # 7: RESULTS_DIR + 'results-gamma-varying-model-20jobs-rate-005-cv-2.csv'
    }
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if 1 in exps:
        # EXP #1 - experiments on poisson distribution with varying arrival rates
        distribution = 'poisson'
        model = 'vicuna-13b'
        df = pd.DataFrame(columns=['arrival_rate', 'algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for avg_arrival_rate in arrival_rates_poisson:
            print('Avg arrival rate =', avg_arrival_rate)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(NUM_JOBS, distribution, avg_arrival_rate, algorithms,
                                 model=model, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'arrival_rate': avg_arrival_rate, 'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'arrival_rate': avg_arrival_rate, 'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[1])
    if 2 in exps:
        # EXP #2 - experiments on poisson distribution with varying number of jobs
        distribution = 'poisson'
        avg_arrival_rate = 0.5
        model = 'vicuna-13b'
        df = pd.DataFrame(columns=['num_jobs', 'algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for num_jobs in num_jobs_list:
            print('Number of jobs =', num_jobs)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(num_jobs, distribution, avg_arrival_rate, algorithms,
                                 model=model, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'num_jobs': num_jobs, 'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'num_jobs': num_jobs, 'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[2])
    if 3 in exps:
        # EXP #3 - experiments on gamma distribution with varying arrival rates
        distribution = 'gamma'
        cv_arrival_rate = 2
        model = 'vicuna-13b'
        df = pd.DataFrame(columns=['arrival_rate', 'algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for avg_arrival_rate in arrival_rates_gamma:
            print('Avg arrival rate =', avg_arrival_rate)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(NUM_JOBS, distribution, avg_arrival_rate, algorithms, model=model,
                                 cv_arrival_rate=cv_arrival_rate, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'arrival_rate': avg_arrival_rate, 'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'arrival_rate': avg_arrival_rate, 'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[3])
    if 4 in exps:
        # EXP #4 - experiments on gamma distribution with varying CVs
        distribution = 'gamma'
        avg_arrival_rate = 0.03
        model = 'vicuna-13b'
        df = pd.DataFrame(columns=['cv', 'algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for cv_arrival_rate in cv_arrival_rates_gamma:
            print('CV =', cv_arrival_rate)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(NUM_JOBS, distribution, avg_arrival_rate, algorithms, model=model,
                                 cv_arrival_rate=cv_arrival_rate, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'cv': cv_arrival_rate, 'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'cv': cv_arrival_rate, 'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[4])
    if 5 in exps:
        # EXP #5 - experiments on gamma distribution with varying number of jobs
        distribution = 'gamma'
        avg_arrival_rate = 0.01
        cv_arrival_rate = 2
        model = 'vicuna-13b'
        df = pd.DataFrame(columns=['num_jobs', 'algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for num_jobs in num_jobs_list:
            print('Number of jobs =', num_jobs)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(num_jobs, distribution, avg_arrival_rate, algorithms, model=model,
                                 cv_arrival_rate=cv_arrival_rate, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'num_jobs': num_jobs, 'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'num_jobs': num_jobs, 'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[5])
    if 6 in exps:
        # EXP #6 - experiments on poisson distribution with different models
        distribution = 'poisson'
        avg_arrival_rate = 1
        num_jobs = 20
        df = pd.DataFrame(columns=['algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for model in ['vicuna-33b', 'vicuna-13b', 'vicuna-7b', 'palm-2', 'gpt-4', 'gpt-3.5-turbo', 'gpt4all-13b-snoozy',
                      'llama-2-13b-chat', 'llama-2-7b-chat', 'llama-13b', 'claude-instant-1', 'claude-1', 'claude-2',
                      'alpaca-13b', 'fastchat-t5-3b']:
            print('Model:', model)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(num_jobs, distribution, avg_arrival_rate, algorithms,
                                 model=model, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[6])
    if 7 in exps:
        # EXP #7 - experiments on gamma distribution with different models
        distribution = 'gamma'
        avg_arrival_rate = 0.05
        cv_arrival_rate = 2
        num_jobs = 20
        df = pd.DataFrame(columns=['algorithm', 'waiting_time', 'jct', 'throughput', 'model'])
        for model in ['vicuna-33b', 'vicuna-13b', 'vicuna-7b', 'palm-2', 'gpt-4', 'gpt-3.5-turbo', 'gpt4all-13b-snoozy',
                      'llama-2-13b-chat', 'llama-2-7b-chat', 'llama-13b', 'claude-instant-1', 'claude-1', 'claude-2',
                      'alpaca-13b', 'fastchat-t5-3b']:
            print('Model:', model)
            wait_time_list_map = init_map(algorithms)
            jct_list_map = init_map(algorithms)
            throughput_list_map = init_map(algorithms)
            p99_jct_list_map = init_map(algorithms)
            for _ in range(NUM_EXP_RUNS):
                result = run_exp(num_jobs, distribution, avg_arrival_rate, algorithms, model=model,
                                 cv_arrival_rate=cv_arrival_rate, data_path=LLM_DATA_PATH, classifiers=CLASSIFIERS)
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        wait_time_list_map[algo].append(result[algo][0])
                        jct_list_map[algo].append(result[algo][1])
                        throughput_list_map[algo].append(result[algo][2])
                        p99_jct_list_map[algo].append(result[algo][3])
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            wait_time_list_map[algo+'-'+classifier].append(result[algo][classifier][0])
                            jct_list_map[algo+'-'+classifier].append(result[algo][classifier][1])
                            throughput_list_map[algo+'-'+classifier].append(result[algo][classifier][2])
                            p99_jct_list_map[algo+'-'+classifier].append(result[algo][classifier][3])
                new_rows = []
                for algo in algorithms:
                    if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                        new_rows.append({
                            'algorithm': algo, 'model': model,
                            'waiting_time': result[algo][0],
                            'jct': result[algo][1],
                            'throughput': result[algo][2],
                            'p99_jct': result[algo][3]})
                    else:
                        # there are multiple predictors in sjfp
                        for classifier in CLASSIFIERS:
                            new_rows.append({
                                'algorithm': algo+'-'+classifier, 'model': model,
                                'waiting_time': result[algo][classifier][0],
                                'jct': result[algo][classifier][1],
                                'throughput': result[algo][classifier][2],
                                'p99_jct': result[algo][classifier][3]})
                df = pd.concat([df, pd.DataFrame(new_rows)])
            for algo in algorithms:
                if algo != 'sjfp' or len(CLASSIFIERS) == 1:
                    print(algo.upper(), 'Avg waiting time:', np.mean(wait_time_list_map[algo]))
                    print(algo.upper(), 'Avg JCT:', np.mean(jct_list_map[algo]))
                    print(algo.upper(), 'p99 JCT:', np.mean(p99_jct_list_map[algo]))
                    print(algo.upper(), 'Avg throughput:', np.mean(throughput_list_map[algo]))
                else:
                    # there are multiple predictors in sjfp
                    for classifier in CLASSIFIERS:
                        print(algo.upper()+'-'+classifier, 'Avg waiting time:', np.mean(wait_time_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg JCT:', np.mean(jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'p99 JCT:', np.mean(p99_jct_list_map[algo+'-'+classifier]))
                        print(algo.upper()+'-'+classifier, 'Avg throughput:', np.mean(throughput_list_map[algo+'-'+classifier]))
            # print(df)
            # print(len(df))
            df.to_csv(exps[7])


if __name__ == '__main__':
    main()
