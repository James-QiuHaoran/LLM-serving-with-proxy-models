import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from job import create_jobs, create_jobs_from_llm_data
from fcfs import FCFS
from sjf import SJF


LLM_DATA_PATH = 'prediction/regression_predictions_vicuna.csv'
VERBOSE = False


class Simulator:
    @staticmethod
    def calculate_avg_wait_duration(job_list):
        """
        This method calculates average waiting time for a given completed job list.
        :param job_list:Job
        :return avg_wait_duration:double
        """
        total_wait_duration = 0
        for job in job_list:
            total_wait_duration += job.get_waiting_duration()

        avg_wait_duration = total_wait_duration / len(job_list)
        return avg_wait_duration

    @staticmethod
    def get_completion_time(job_list):
        """
        This method returns the total completion time for a given completed job list.
        :param job_list:Job
        :return completion_time:double
        """
        return max([job.get_completion_time() for job in job_list])
    
    @staticmethod
    def get_throughput(job_list, percentage=25):
        """
        This method returns the throughput for a given completed job list.
        :param job_list:Job
        :return throughput:double
        """
        # Check the real-time throughput when finishing 5%, 10%, 25%, 50%, 75%, 90%, 100% of the jobs
        job_finish_times = [job.get_completion_time() for job in job_list]
        job_finish_times = sorted(job_finish_times)
        # for percentage in [5, 10, 25, 50, 75, 90, 100]:
        #     pos = int(np.percentile(list(range(len(job_finish_times))), percentage))
        #     print('Throughput at '+str(percentage)+'%:', pos / job_finish_times[pos])
        pos = int(np.percentile(list(range(len(job_finish_times))), percentage))
        # print('Throughput at '+str(percentage)+'%:', pos / job_finish_times[pos])
        return pos / job_finish_times[pos]

    @staticmethod
    def calculate_per_job_avg_latency(job_list):
        """
        This method calculates the average per-job latency (= waiting time + execution time) for a given completed job
        list.
        :param job_list:Job
        :return: avg_latency:double
        """
        total_latency = 0
        for job in job_list:
            # total_latency += job.get_waiting_duration() + job.get_execution_duration()
            total_latency += job.get_completion_time() - job.get_arrival_time()

        avg_latency = total_latency / len(job_list)
        return avg_latency

    @staticmethod
    def calculate_percentile_latency(job_list, percentile=99):
        """
        This method calculates the percentile latency (= waiting time + execution time) for a given completed job
        list.
        :param job_list:Job
        :return: latency:double
        """
        # latency_list = [job.get_waiting_duration() + job.get_execution_duration() for job in job_list]
        latency_list = [job.get_completion_time() - job.get_arrival_time() for job in job_list]
        return np.percentile(latency_list, percentile)

    @staticmethod
    def print_job_list(job_list):
        """
        This method prints information for a given completed job list.
        :param job_list:Job
        """
        for job in job_list:
            print("JobId:", job.get_job_id(),
                  "Execution time:", job.get_execution_duration(),
                  "Arrival Time:", job.get_arrival_time())
            print("Completion time:", job.get_completion_time())
            print("Waiting time:", job.get_waiting_duration())
            print("-----------------------------------------------------------------")

    @staticmethod
    def plot_job_completion_figure(map_job_lists):
        """
        This method visualizes the job completion on a figure where X-axis is time and Y-axis is the number of completed
        jobs, given the completed job list as the input.
        :param map_job_lists:Map<String,Job>
        """
        for key in map_job_lists:
            job_list = map_job_lists[key]
            jobs_x = []
            jobs_y = []
            num_completed_jobs = 0
            for job in job_list:
                num_completed_jobs += 1
                jobs_x.append(job.get_completion_time())
                jobs_y.append(num_completed_jobs)
            plt.plot(jobs_x, jobs_y, label=key)
        plt.xlabel('Time')
        plt.ylabel('Completed Job Count')
        plt.xlim(0)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


def main():
    print("Starting Simulator...")
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument("-n", "--num_jobs", metavar="NUM_JOBS",
                               help="Number of jobs to be scheduled on a scheduling algorithm",
                               type=int, required=True, action="store")
    required_args.add_argument("-a", "--algorithm", metavar="ALGO",
                               help="Scheduling algorithm (fcfs, sjf, sjfp, all)",
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
        sim = Simulator()

        # create jobs
        if use_llm:
            job_list = create_jobs_from_llm_data(num_jobs, model=model,
                                                 distribution=distribution, arrival_rate=avg_arrival_rate,
                                                 std=std_arrival_rate, coefficient_of_variance=cv_arrival_rate,
                                                 data_path=LLM_DATA_PATH,
                                                 per_token_latency=per_token_latency, const_latency=const_latency)
        else:
            job_list = create_jobs(num_jobs, distribution=distribution, arrival_rate=avg_arrival_rate,
                                   std=std_arrival_rate, coefficient_of_variance=cv_arrival_rate)
        print('Created jobs with', distribution, 'distribution at', avg_arrival_rate, 'avg requests/sec')
        map_of_completed_job_lists = {}

        if algorithm == 'fcfs' or algorithm == 'all':
            fcfs_job_list = deepcopy(job_list)
            scheduler_fcfs = FCFS()
            print('Started job execution through FCFS Scheduling')
            fcfs_job_list = scheduler_fcfs.schedule(fcfs_job_list)
            print('Finished job execution through FCFS Scheduling')
            if VERBOSE:
                sim.print_job_list(fcfs_job_list)
            print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(fcfs_job_list))
            print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(fcfs_job_list))
            print('>>>> Final completion time:', sim.get_completion_time(fcfs_job_list))
            print('-----------------------------------------------------------------\n')
            map_of_completed_job_lists['fcfs'] = fcfs_job_list
        if algorithm == 'sjf' or algorithm == 'all':
            sjf_job_list = deepcopy(job_list)
            scheduler_sjf = SJF()
            print('Started job execution through SJF Scheduling')
            sjf_job_list = scheduler_sjf.schedule(sjf_job_list)
            print('Finished job execution through SJF Scheduling')
            if VERBOSE:
                sim.print_job_list(sjf_job_list)
            print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(sjf_job_list))
            print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(sjf_job_list))
            print('>>>> Final completion time:', sim.get_completion_time(sjf_job_list))
            print('-----------------------------------------------------------------\n')
            map_of_completed_job_lists['sjf'] = sjf_job_list
        if algorithm == 'sjfp' or algorithm == 'all':
            sjfp_job_list = deepcopy(job_list)
            scheduler_sjf = SJF(use_prediction=True)
            print('Started job execution through SJF-prediction Scheduling')
            sjfp_job_list = scheduler_sjf.schedule(sjfp_job_list)
            print('Finished job execution through SJF-predictioin Scheduling')
            if VERBOSE:
                sim.print_job_list(sjfp_job_list)
            print('>>>> Avg waiting duration:', sim.calculate_avg_wait_duration(sjfp_job_list))
            print('>>>> Avg latency:', sim.calculate_per_job_avg_latency(sjfp_job_list))
            print('>>>> Final completion time:', sim.get_completion_time(sjfp_job_list))
            print('-----------------------------------------------------------------\n')
            map_of_completed_job_lists['sjfp'] = sjfp_job_list

        # visualization
        sim.plot_job_completion_figure(map_of_completed_job_lists)

    print('Simulator Exiting...')


if __name__ == '__main__':
    main()
