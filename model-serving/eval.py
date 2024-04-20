import argparse
from copy import deepcopy

from job import create_jobs, create_jobs_from_llm_data
from fcfs import FCFS
from sjf import SJF
from fcfs_dynamic import FCFSWithDyanmicBatching
from sjf_dynamic import SJFWithDynamicBatching
from simulator import Simulator
from util import *


LLM_DATA_PATH = 'prediction/regression_predictions_vicuna.csv'


def run_exp(num_jobs, distribution, avg_arrival_rate, algorithms, use_llm=True, model='vicuna-13b',
            std_arrival_rate=1, cv_arrival_rate=2, per_token_latency=0.02, const_latency=0.1,
            data_path=LLM_DATA_PATH, visualization=False, classifiers=['predictor']):
    # create the simulator
    sim = Simulator()

    # create jobs
    if use_llm:
        # for a single predictor used in sjfp
        if len(classifiers) == 1:
            job_list = create_jobs_from_llm_data(num_jobs, model=model,
                                                distribution=distribution, arrival_rate=avg_arrival_rate,
                                                std=std_arrival_rate, coefficient_of_variance=cv_arrival_rate,
                                                data_path=data_path,
                                                per_token_latency=per_token_latency, const_latency=const_latency)
        else:
            # for multiple predictors used in sjfp
            job_dict = create_jobs_from_llm_data(num_jobs, model=model,
                                                distribution=distribution, arrival_rate=avg_arrival_rate,
                                                std=std_arrival_rate, coefficient_of_variance=cv_arrival_rate,
                                                data_path=data_path,
                                                per_token_latency=per_token_latency, const_latency=const_latency,
                                                return_dict=True)
            job_list = job_dict[classifiers[0]]  # for FCFS and SJF
    else:
        job_list = create_jobs(num_jobs, distribution=distribution, arrival_rate=avg_arrival_rate,
                               std=std_arrival_rate, coefficient_of_variance=cv_arrival_rate)
    print('Created jobs with', distribution, 'distribution at', avg_arrival_rate, 'avg requests/sec')

    map_of_completed_job_lists = {}
    ret_fcfs = []
    ret_sjf = []
    ret_sjfp = []
    if len(classifiers) > 1:
        ret_sjfp = {}
    if 'fcfs' in algorithms:
        print('FCFS')
        fcfs_job_list = deepcopy(job_list)
        if DYNAMIC_BATCHING:
            scheduler_fcfs = FCFSWithDyanmicBatching()
        else:
            scheduler_fcfs = FCFS()
        fcfs_job_list = scheduler_fcfs.schedule(fcfs_job_list)
        # sim.print_job_list(fcfs_job_list)
        print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(fcfs_job_list))
        print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(fcfs_job_list))
        print('>>>> p99 latency:', sim.calculate_percentile_latency(fcfs_job_list, 99))
        print('>>>> Final completion time:', sim.get_completion_time(fcfs_job_list), len(fcfs_job_list), 'jobs')
        # print('>>>> Avg throughput:', len(fcfs_job_list) / sim.get_completion_time(fcfs_job_list))
        print('>>>> Throughput:', sim.get_throughput(fcfs_job_list))
        print('-----------------------------------------------------------------\n')
        map_of_completed_job_lists['fcfs'] = fcfs_job_list
        ret_fcfs.append(sim.calculate_avg_wait_duration(fcfs_job_list))
        ret_fcfs.append(sim.calculate_per_job_avg_latency(fcfs_job_list))
        # ret_fcfs.append(len(fcfs_job_list) / sim.get_completion_time(fcfs_job_list))
        ret_fcfs.append(sim.get_throughput(fcfs_job_list))
        ret_fcfs.append(sim.calculate_percentile_latency(fcfs_job_list, 99))
    if 'sjf' in algorithms:
        print('SJF')
        sjf_job_list = deepcopy(job_list)
        if DYNAMIC_BATCHING:
            scheduler_sjf = SJFWithDynamicBatching()
        else:
            scheduler_sjf = SJF()
        sjf_job_list = scheduler_sjf.schedule(sjf_job_list)
        # sim.print_job_list(sjf_job_list)
        print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(sjf_job_list))
        print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(sjf_job_list))
        print('>>>> p99 latency:', sim.calculate_percentile_latency(sjf_job_list, 99))
        print('>>>> Final completion time:', sim.get_completion_time(sjf_job_list), len(sjf_job_list), 'jobs')
        # print('>>>> Avg throughput:', len(sjf_job_list) / sim.get_completion_time(sjf_job_list))
        print('>>>> Throughput:', sim.get_throughput(sjf_job_list))
        print('-----------------------------------------------------------------\n')
        map_of_completed_job_lists['sjf'] = sjf_job_list
        ret_sjf.append(sim.calculate_avg_wait_duration(sjf_job_list))
        ret_sjf.append(sim.calculate_per_job_avg_latency(sjf_job_list))
        # ret_sjf.append(len(sjf_job_list) / sim.get_completion_time(sjf_job_list))
        ret_sjf.append(sim.get_throughput(sjf_job_list))
        ret_sjf.append(sim.calculate_percentile_latency(sjf_job_list, 99))
    if 'sjfp' in algorithms:
        print('SJF with predictions')
        if len(classifiers) == 1:
            sjfp_job_list = deepcopy(job_list)
            if DYNAMIC_BATCHING:
                scheduler_sjf = SJFWithDynamicBatching(use_prediction=True)
            else:
                scheduler_sjf = SJF(use_prediction=True)
            sjfp_job_list = scheduler_sjf.schedule(sjfp_job_list)
            # sim.print_job_list(sjfp_job_list)
            print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(sjfp_job_list))
            print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(sjfp_job_list))
            print('>>>> p99 latency:', sim.calculate_percentile_latency(sjfp_job_list, 99))
            print('>>>> Final completion time:', sim.get_completion_time(sjfp_job_list), len(sjfp_job_list), 'jobs')
            # print('>>>> Avg throughput:', len(sjfp_job_list) / sim.get_completion_time(sjfp_job_list))
            print('>>>> Throughput:', sim.get_throughput(sjfp_job_list))
            print('-----------------------------------------------------------------\n')
            map_of_completed_job_lists['sjf-prediction'] = sjfp_job_list
            ret_sjfp.append(sim.calculate_avg_wait_duration(sjfp_job_list))
            ret_sjfp.append(sim.calculate_per_job_avg_latency(sjfp_job_list))
            # ret_sjfp.append(len(sjfp_job_list) / sim.get_completion_time(sjfp_job_list))
            ret_sjfp.append(sim.get_throughput(sjfp_job_list))
            ret_sjfp.append(sim.calculate_percentile_latency(sjfp_job_list, 99))
        else:
            for classifier in classifiers:
                print('SJFP with predictor:', classifier)
                sjfp_job_list = deepcopy(job_dict[classifier])
                if DYNAMIC_BATCHING:
                    scheduler_sjf = SJFWithDynamicBatching(use_prediction=True)
                else:
                    scheduler_sjf = SJF(use_prediction=True)
                sjfp_job_list = scheduler_sjf.schedule(sjfp_job_list)
                # sim.print_job_list(sjfp_job_list)
                print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(sjfp_job_list))
                print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(sjfp_job_list))
                print('>>>> p99 latency:', sim.calculate_percentile_latency(sjfp_job_list, 99))
                print('>>>> Final completion time:', sim.get_completion_time(sjfp_job_list), len(sjfp_job_list), 'jobs')
                # print('>>>> Avg throughput:', len(sjfp_job_list) / sim.get_completion_time(sjfp_job_list))
                print('>>>> Throughput:', sim.get_throughput(sjfp_job_list))
                print('-----------------------------------------------------------------\n')
                map_of_completed_job_lists['sjfp-'+classifier] = sjfp_job_list
                ret_sjfp_cls = []
                ret_sjfp_cls.append(sim.calculate_avg_wait_duration(sjfp_job_list))
                ret_sjfp_cls.append(sim.calculate_per_job_avg_latency(sjfp_job_list))
                # ret_sjfp_cls.append(len(sjfp_job_list) / sim.get_completion_time(sjfp_job_list))
                ret_sjfp_cls.append(sim.get_throughput(sjfp_job_list))
                ret_sjfp_cls.append(sim.calculate_percentile_latency(sjfp_job_list, 99))
                ret_sjfp[classifier] = ret_sjfp_cls

    # visualization
    if visualization:
        sim.plot_job_completion_figure(map_of_completed_job_lists)

    return {
        'fcfs': ret_fcfs,
        'sjf': ret_sjf,
        'sjfp': ret_sjfp
    }


def main():
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument("-e", "--num_experiments", metavar="NUM_EXP",
                               help="Number of experiments to run",
                               type=int, required=True, action="store")
    required_args.add_argument("-n", "--num_jobs", metavar="NUM_JOBS",
                               help="Number of jobs to be scheduled on a scheduling algorithm",
                               type=int, required=True, action="store")
    required_args.add_argument("-a", "--algorithm", metavar="ALGO",
                               help="Scheduling algorithm (fcfs, sjf, sjfp)",
                               type=str, required=True, action="store")
    required_args.add_argument("-d", "--distribution", metavar="DISTRIBUTION",
                               help="Job arrival distribution (poisson, uniform, gamma)",
                               type=str, required=True, action="store")
    required_args.add_argument("-r", "--rate", metavar="RATE",
                               help="Average job arrival rate",
                               type=float, required=False, default=1, action="store")
    required_args.add_argument("-s", "--std", metavar="STD",
                               help="Standard deviation of job arrival rate",
                               type=float, required=False, default=1, action="store")
    required_args.add_argument("-c", "--cv", metavar="COEFFICIENT_OF_VARIATION",
                               help="Coefficient of variation of job arrival rate",
                               type=float, required=False, default=2, action="store")
    required_args.add_argument("--llm",
                               help="Use LLM data for execution time sampling or not",
                               required=False, default=False, action="store_true")
    required_args.add_argument("-m", "--model", metavar="MODEL",
                               help="LLM model to get traces for",
                               type=str, required=False, default='vicuna-13b', action="store")
    required_args.add_argument("--per_token_latency", metavar="LATENCY",
                               help="Per-token generation latency, default 0.02",
                               type=float, required=False, default=0.02, action="store")
    required_args.add_argument("--const_latency", metavar="LATENCY",
                               help="Constant latency (e.g., DNS lookup), default 0.1",
                               type=float, required=False, default=0.1, action="store")

    args = parser.parse_args()

    num_exp = args.num_experiments
    num_jobs = args.num_jobs
    algorithm = args.algorithm
    distribution = args.distribution
    avg_arrival_rate = args.rate
    std_arrival_rate = args.std
    cv_arrival_rate = args.cv
    use_llm = args.llm
    model = args.model
    per_token_latency = args.per_token_latency
    const_latency = args.const_latency

    if num_jobs <= 0:
        print('Number of jobs to schedule should be > 0!')
        exit()
    elif algorithm not in ['fcfs', 'sjf', 'sjfp', 'all']:
        print('Scheduling algorithms should be one of the [fcfs, sjf, sjfp, all]!')
        exit()
    elif distribution not in ['poisson', 'uniform', 'gamma']:
        print('Scheduling algorithms should be one of the [poisson, uniform, gamma]!')
        exit()
    else:
        algorithms = [algorithm]
        if algorithm == 'all':
            algorithms = ['fcfs', 'sjf', 'sjfp']
        for i in range(num_exp):
            run_exp(num_jobs, distribution, avg_arrival_rate, algorithms,
                    std_arrival_rate=std_arrival_rate, cv_arrival_rate=cv_arrival_rate,
                    use_llm=use_llm, model=model,
                    per_token_latency=per_token_latency, const_latency=const_latency, visualization=True)


if __name__ == '__main__':
    main()
