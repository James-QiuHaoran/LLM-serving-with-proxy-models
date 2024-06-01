import numpy as np
import pandas as pd
import random

from util import *


LABEL_COLUMN_NAME = 'output_token_length'
PREDICTION_COLUMN_NAME = 'predicted_length'
MODEL_COLUMN_NAME = 'model'
PREDICTION_POSTFIX = '_prediction'
AZURE_TRACES_DATA_PATH_CODE = 'traces/AzureLLMInferenceTrace_code_int.csv'
AZURE_TRACES_DATA_PATH_CONV = 'traces/AzureLLMInferenceTrace_conv_int.csv'


np.random.seed(42)
random.seed(42)


class Job:
    def __init__(self, job_id, execution_duration, arrival_time, predicted_execution_duration=None, status='CREATED'):
        self.job_id = job_id
        self.status = status
        self.arrival_time = arrival_time  # timestamp that the job arrives
        self.execution_duration = execution_duration
        self.waiting_duration = None
        self.completion_time = None  # timestamp that the job is finished
        self.predicted_execution_duration = predicted_execution_duration
        self.curr_waiting_time = 0  # used to avoid starvation

    def get_job_id(self):
        return self.job_id

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_arrival_time(self):
        return self.arrival_time

    def set_waiting_duration(self, waiting_duration):
        self.waiting_duration = waiting_duration

    def get_waiting_duration(self):
        return self.waiting_duration

    def set_execution_duration(self, execution_duration):
        self.execution_duration = execution_duration

    def get_execution_duration(self):
        return self.execution_duration

    def set_completion_time(self, completion_time):
        self.completion_time = completion_time

    def get_completion_time(self):
        return self.completion_time
    
    def set_predicted_execution_duration(self, predicted_duration):
        self.predicted_execution_duration = predicted_duration
    
    def get_predicted_execution_duration(self):
        return self.predicted_execution_duration

    def print_info(self):
        print('ID:', self.job_id, 'status:', self.status)
        print('Arrival time:', self.arrival_time,
              'Execution duration:', self.execution_duration,
              'Predicted Exec duration:', self.predicted_execution_duration,
              'Waiting duration:', self.waiting_duration,
              'Completion time:', self.completion_time)


def generate_arrival_list(request_list, total_num_requests):
    """
    This method generate a list of arrival times for each request given a request_list where each element at index i in
    the request_list indicates the number of requests arrive at time step i.
    :param request_list:int
    :param total_num_requests:int
    :return arrival_list:int
    """
    arrival_list = []
    current_num_requests = 0
    for i, num_requests in enumerate(request_list):
        for j in range(num_requests):
            arrival_list.append(i)
            current_num_requests += 1
            if current_num_requests == total_num_requests:
                return arrival_list
    return arrival_list


def create_jobs(num_jobs, arrival_rate=1, std=1, coefficient_of_variance=3, distribution='poisson', trace_scale=1.0):
    """
    This method creates a series of jobs based on the specified average arrival rate, standard deviation, and the
    distribution for simulating the job arrival process.
    :param num_jobs:int
    :param arrival_rate:double
    :param std:double
    :param coefficient_of_variance:double
    :param distribution:str
    :param trace_scale:double
    :return job_list:Job
    """
    job_list = []
    if distribution == 'poisson':
        request_list = np.random.poisson(arrival_rate, int(num_jobs/arrival_rate))
        while sum(request_list) < num_jobs:
            request_list = np.append(request_list, np.random.poisson(arrival_rate, int(num_jobs/arrival_rate)))
        print('Request list:', request_list)
        arrival_times = generate_arrival_list(request_list, num_jobs)
    elif distribution == 'uniform':
        intervals = np.random.uniform(1/(arrival_rate-std/2), 1/(arrival_rate+std/2), num_jobs)
        arrival_times = np.cumsum(np.sort(intervals))
    elif distribution == 'gamma':
        shape = (1 / coefficient_of_variance) ** 2
        scale = coefficient_of_variance ** 2 / arrival_rate
        intervals = np.random.gamma(shape, scale, num_jobs)
        arrival_times = np.cumsum(np.sort(intervals))
    elif distribution == 'azure-conv':
        # replay azure LLM inference traces
        # read from CSV file
        df = pd.read_csv(AZURE_TRACES_DATA_PATH_CONV)
        # random pick a random number between 0 and len(df)-num_jobs
        start_index = random.randint(0, len(df) - num_jobs)
        # select num_jobs rows from start_index to (start_index+num_jobs)
        df = df.iloc[start_index:start_index+num_jobs]
        # convert df['TIMESTAMP'] to arrival_times
        arrival_times = pd.to_datetime(df['TIMESTAMP']).astype(int)
        arrival_times = np.array(arrival_times)
        # deduct each arrival_time by the first arrival_time to get relative arrival times
        arrival_times = arrival_times - arrival_times[0]
        # apply the trace_scale to intensify the arrival times
        arrival_times = np.round(arrival_times / trace_scale, 1)
    elif distribution == 'azure-code':
        # replay azure LLM inference traces
        # read from CSV file
        df = pd.read_csv(AZURE_TRACES_DATA_PATH_CODE)
        # random pick a random number between 0 and len(df)-num_jobs
        start_index = random.randint(0, len(df) - num_jobs)
        # select num_jobs rows from start_index to (start_index+num_jobs)
        df = df.iloc[start_index:start_index+num_jobs]
        # convert df['TIMESTAMP'] to arrival_times
        arrival_times = pd.to_datetime(df['TIMESTAMP']).astype(int)
        arrival_times = np.array(arrival_times)
        # deduct each arrival_time by the first arrival_time to get relative arrival times
        arrival_times = arrival_times - arrival_times[0]
        # apply the trace_scale to intensify the arrival times
        arrival_times = np.round(arrival_times / trace_scale, 1)
    else:
        print('Distributions should be chosen from [poisson, uniform, gamma, azure-code, azure-conv]!')
        exit(0)
    print("Generated arrival times:", arrival_times)

    for job_id in range(num_jobs):
        exec_time = random.randint(1, 10)
        arrival_time = arrival_times[job_id]
        job_list.append(Job(job_id, exec_time, arrival_time))

    return job_list


def create_jobs_from_llm_data(num_jobs, arrival_rate=1, std=1, coefficient_of_variance=3, distribution='poisson',
                              model='vicuna-13b', per_token_latency=0.02, const_latency=0.1,
                              data_path='../characterization/token_lengths_all.csv', return_dict=False,
                              per_round_eval=PER_ROUND_EVAL, turn_id=TURN_ID, trace_scale=1.0):
    """
    This method creates a series of jobs based on the specified average arrival rate, standard deviation, and the
    distribution for simulating the job arrival process.
    :param num_jobs:int
    :param arrival_rate:double
    :param std:double
    :param coefficient_of_variance:double
    :param distribution:str
    :param model:str
    :param per_token_latency:double
    :param const_latency:double
    :param data_path:str
    :param trace_scale:double
    :return job_list:Job
    """
    if distribution == 'poisson':
        request_list = np.random.poisson(arrival_rate, int(num_jobs / arrival_rate))
        while sum(request_list) < num_jobs:
            request_list = np.append(request_list, np.random.poisson(arrival_rate, int(num_jobs / arrival_rate)))
        # print('Request list:', request_list)
        arrival_times = generate_arrival_list(request_list, num_jobs)
    elif distribution == 'uniform':
        intervals = np.random.uniform(1 / (arrival_rate - std / 2), 1 / (arrival_rate + std / 2), num_jobs)
        arrival_times = np.cumsum(np.sort(intervals))
    elif distribution == 'gamma':
        shape = (1 / coefficient_of_variance) ** 2
        scale = coefficient_of_variance ** 2 / arrival_rate
        intervals = np.random.gamma(shape, scale, num_jobs)
        arrival_times = np.cumsum(np.sort(intervals))
    elif distribution == 'azure-conv':
        # replay azure LLM inference traces
        # read from CSV file
        df = pd.read_csv(AZURE_TRACES_DATA_PATH_CONV)
        # random pick a random number between 0 and len(df)-num_jobs
        start_index = random.randint(0, len(df) - num_jobs)
        # select num_jobs rows from start_index to (start_index+num_jobs)
        df = df.iloc[start_index:start_index+num_jobs]
        # convert df['TIMESTAMP'] to arrival_times
        arrival_times = pd.to_datetime(df['TIMESTAMP']).astype(int)
        arrival_times = np.array(arrival_times)
        # deduct each arrival_time by the first arrival_time to get relative arrival times
        arrival_times = arrival_times - arrival_times[0]
        # apply the trace_scale to intensify the arrival times
        arrival_times = np.round(arrival_times / trace_scale, 1)
    elif distribution == 'azure-code':
        # replay azure LLM inference traces
        # read from CSV file
        df = pd.read_csv(AZURE_TRACES_DATA_PATH_CODE)
        # random pick a random number between 0 and len(df)-num_jobs
        start_index = random.randint(0, len(df) - num_jobs)
        # select num_jobs rows from start_index to (start_index+num_jobs)
        df = df.iloc[start_index:start_index+num_jobs]
        # convert df['TIMESTAMP'] to arrival_times
        arrival_times = pd.to_datetime(df['TIMESTAMP']).astype(int)
        arrival_times = np.array(arrival_times)
        # deduct each arrival_time by the first arrival_time to get relative arrival times
        arrival_times = arrival_times - arrival_times[0]
        # apply the trace_scale to intensify the arrival times
        arrival_times = np.round(arrival_times / trace_scale, 1)
    else:
        print('Distributions should be chosen from [poisson, uniform, gamma, azure-code, azure-conv]!')
        exit(0)
    # print("Generated arrival times:", arrival_times)

    # load LLM data
    df = pd.read_csv(data_path)
    if per_round_eval:
        # filter data by turn_id
        df = df[df[TURN_ID_COLUMN_NAME] == turn_id]
    if model not in df[MODEL_COLUMN_NAME].unique():
        print('Model', model, 'not found in the dataset!\nPlease select from:', list(df[MODEL_COLUMN_NAME].unique()))
        exit()

    print('Sampling data from traces for model', model)
    # list_num_tokens = list(df[df[MODEL_COLUMN_NAME] == model].sample(n=num_jobs)[LABEL_COLUMN_NAME])
    samples_df_rows = df[df[MODEL_COLUMN_NAME] == model].sample(n=num_jobs)

    if not return_dict:
        # generate a list of jobs and return that list
        job_list = []
        for job_id in range(num_jobs):
            # get the actual execution time from the LLM data
            exec_time = round(const_latency + samples_df_rows.iloc[job_id][LABEL_COLUMN_NAME] * per_token_latency, 2)
            arrival_time = arrival_times[job_id]
            # get the estimated execution time from the predictor
            if PREDICTION_COLUMN_NAME in samples_df_rows.columns:
                predicted_exec_time = round(const_latency + samples_df_rows.iloc[job_id][PREDICTION_COLUMN_NAME] * per_token_latency, 2)
                # predicted_exec_time = round(
                #     const_latency + random.randint(0, 4) * per_token_latency, 2)  # a random token length predictor
                job_list.append(Job(job_id, exec_time, arrival_time, predicted_execution_duration=predicted_exec_time))
            else:
                job_list.append(Job(job_id, exec_time, arrival_time))

        return job_list
    else:
        # generate a dictionary of jobs and return that dictionary
        # dict key: predictor_name --> predictor used for predicting job exec time
        # dict value: a list of jobs
        dict_job_list = {}
        for classifier in CLASSIFIERS:
            job_list = []
            for job_id in range(num_jobs):
                # get the actual execution time from the LLM data
                exec_time = round(const_latency + samples_df_rows.iloc[job_id][LABEL_COLUMN_NAME] * per_token_latency, 2)
                arrival_time = arrival_times[job_id]
                # get the estimated execution time from the LLM data
                prediction_column = classifier + PREDICTION_POSTFIX
                if prediction_column in samples_df_rows.columns:
                    predicted_exec_time = round(const_latency + samples_df_rows.iloc[job_id][prediction_column] * per_token_latency, 2)
                    # predicted_exec_time = round(
                    #     const_latency + random.randint(0, 4) * per_token_latency, 2)  # a random token length predictor
                    job_list.append(Job(job_id, exec_time, arrival_time, predicted_execution_duration=predicted_exec_time))
                else:
                    job_list.append(Job(job_id, exec_time, arrival_time))

            dict_job_list[classifier] = job_list
        return dict_job_list


if __name__ == '__main__':
    # Test 1
    jobs = create_jobs(10, distribution='gamma', arrival_rate=2, coefficient_of_variance=3)
    # jobs = create_jobs(10, distribution='poisson', arrival_rate=1)
    for job in jobs:
        job.print_info()

    # Test 2
    jobs = create_jobs_from_llm_data(10)
    for job in jobs:
        job.print_info()

    # Test 3
    jobs = create_jobs_from_llm_data(10, data_path='../characterization/token_lengths_all.csv')
    for job in jobs:
        job.print_info()

    # Test 4
    jobs = create_jobs(10, distribution='azure-conv', trace_scale=1)
    for job in jobs:
        job.print_info()
    jobs = create_jobs(10, distribution='azure-code', trace_scale=0.2)
    for job in jobs:
        job.print_info()

    # Test 5
    jobs = create_jobs_from_llm_data(10, data_path='../characterization/token_lengths_all.csv', distribution='azure-conv', trace_scale=1)
    for job in jobs:
        job.print_info()
    jobs = create_jobs_from_llm_data(10, data_path='../characterization/token_lengths_all.csv', distribution='azure-code', trace_scale=0.3)
    for job in jobs:
        job.print_info()