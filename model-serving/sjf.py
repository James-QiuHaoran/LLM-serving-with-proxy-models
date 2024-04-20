from job import Job
from collections import deque

from util import *


class SJF:
    """
    This class is used to calculate the waiting time and total completion time of each job.
    The scheduling algorithm is shortest-job-first where every time the job with the shortest execution time in the
    queue will be scheduled to start executing.

    The scheduler supports static batching and the batch size is determined by the MAX_BATCH_SIZE parameter.
    The scheduler will try to batch incoming jobs into batches with wait timeout of BATCH_WAIT_TIMEOUT. The two parameters
    are defined in the util.py file.
    """
    completed_jobs = []
    completion_time = 0

    def __init__(self, use_prediction=False):
        self.use_prediction = use_prediction

    def sort_job(self, job_list, metric='arrival_time', reverse=False):
        """Sort the job list based on arrival time.
        :param job_list:Job
        :param metric:String
        :return sorted_job_list:Job
        """
        if metric == 'arrival_time':
            sorted_job_list = sorted(job_list, key=lambda x: x.arrival_time, reverse=reverse)
        elif metric == 'exec_time':
            if self.use_prediction:
                sorted_job_list = sorted(job_list, key=lambda x: x.predicted_execution_duration, reverse=reverse)
            else:
                sorted_job_list = sorted(job_list, key=lambda x: x.execution_duration, reverse=reverse)
        elif metric == 'waiting_time':
            sorted_job_list = sorted(job_list, key=lambda x: x.curr_waiting_time, reverse=reverse)
        else:
            print('Unknown sorting metric:', metric)
            exit()
        return sorted_job_list

    def run(self, job_list):
        """Run SJF algorithm on the job list.
        Calculate the completion time of each job and stores calculated completion_time in the job object.
        Assume that the job list has been sorted based on the arrival times.
        :param job_list:Job
        """
        # create a queue for jobs that have arrived but are waiting for execution
        wait_queue = deque()

        self.completion_time = 0
        self.prev_completion_time = 0
        time_slice = 1
        # job_list in sorted order based on arrival times
        curr_job_idx = 0  # the index of the current job that has NOT been added to the queue
        while curr_job_idx < len(job_list)-1 or len(wait_queue) > 0:
            if len(wait_queue) == 0:
                # wait for the job to arrive
                curr_job = job_list[curr_job_idx]
                job_arrival_time = curr_job.get_arrival_time()

                if self.completion_time >= job_arrival_time:
                    wait_queue.append(curr_job)
                    curr_job_idx += 1
                    # check if other job(s) has arrived yet
                    while curr_job_idx < len(job_list):
                        if job_list[curr_job_idx].get_arrival_time() <= self.completion_time:
                            wait_queue.append(job_list[curr_job_idx])
                            curr_job_idx += 1
                        else:
                            break

                # wait until the arrival time has been reached
                while self.completion_time < job_arrival_time:
                    if self.completion_time + time_slice < job_arrival_time:
                        # not arrived yet
                        self.completion_time += time_slice
                        time_slice = 1
                    else:
                        # arrived
                        remaining_slice = 1 - (job_arrival_time - self.completion_time)
                        self.completion_time = job_arrival_time
                        time_slice = remaining_slice
                        wait_queue.append(curr_job)
                        curr_job_idx += 1

                        # check if other job(s) has arrived yet
                        while curr_job_idx < len(job_list):
                            if job_list[curr_job_idx].get_arrival_time() <= self.completion_time:
                                wait_queue.append(job_list[curr_job_idx])
                                curr_job_idx += 1
                            else:
                                break

            # get the job with the shortest execution time from the wait queue
            sorted_job_list = self.sort_job(wait_queue, metric='exec_time')
            # address the starvation problem
            sorted_job_list_waiting = self.sort_job(wait_queue, metric='waiting_time', reverse=True)
            if sorted_job_list_waiting[0].curr_waiting_time > SJF_STARVATION_WAITING_TIME_THRES or sorted_job_list_waiting[0].curr_waiting_time > sorted_job_list_waiting[0].execution_duration * SJF_STARVATION_WAITING_TIME_COEFFICIENT:
                # prioritize the execution of the long waiting job
                # print(sorted_job_list_waiting[0].curr_waiting_time, sorted_job_list_waiting[0].execution_duration)
                # exit()
                wait_queue = self.convert_to_queue(sorted_job_list_waiting)
            else:
                wait_queue = self.convert_to_queue(sorted_job_list)
            
            # > for the no batching case
            # curr_job = wait_queue.popleft()
            # job_duration = curr_job.get_execution_duration()
            # job_arrival_time = curr_job.get_arrival_time()

            # > for the static batching case
            curr_batch = []
            while len(wait_queue) > 0 and len(curr_batch) < MAX_BATCH_SIZE and wait_queue[0].get_arrival_time() <= self.completion_time + BATCH_WAIT_TIMEOUT:
                curr_batch.append(wait_queue.popleft())
            # get the job_arrival_time and job_duration based on the running_batch
            if len(curr_batch) == MAX_BATCH_SIZE:
                job_arrival_time = max([x.get_arrival_time() for x in curr_batch])  # the latest arrival time in the batch
            else:
                job_arrival_time = self.completion_time + BATCH_WAIT_TIMEOUT
            job_duration = max([x.get_execution_duration() for x in curr_batch])  # the longest job duration in the batch

            # enter the running stage
            # > for the no batching case
            # curr_job.set_status("RUNNING")
            # curr_job.set_waiting_duration(self.completion_time - job_arrival_time)
            # > for the static batching case
            for job in curr_batch:
                job.set_status("RUNNING")
                job.set_waiting_duration(self.completion_time - job.get_arrival_time())

            while job_duration > 0:
                job_duration -= time_slice
                if job_duration < 0:
                    # execution done
                    self.completion_time += job_duration + time_slice
                    remaining_slice = abs(job_duration)
                    time_slice = remaining_slice
                else:
                    # still running
                    self.completion_time += time_slice
                    time_slice = 1

                # check if other job(s) has arrived yet
                while curr_job_idx < len(job_list):
                    if job_list[curr_job_idx].get_arrival_time() <= self.completion_time:
                        wait_queue.append(job_list[curr_job_idx])
                        curr_job_idx += 1
                    else:
                        break

            # job completed
            # > for the no batching case
            # curr_job.set_status("COMPLETED")
            # curr_job.set_completion_time(self.completion_time)
            # self.completed_jobs.append(curr_job)
            # > for the static batching case
            for job in curr_batch:
                job.set_status("COMPLETED")
                job.set_completion_time(self.completion_time)
                self.completed_jobs.append(job)

            # increase the curr_waiting_time of all jobs in the queue
            increment = self.completion_time - self.prev_completion_time
            self.prev_completion_time = self.completion_time
            for job in wait_queue:
                job.curr_waiting_time += increment
        print("Completion time after executing all", curr_job_idx+1, "jobs:", self.completion_time)

    @staticmethod
    def convert_to_queue(job_list):
        """Convert the given job list to queue.
        :param job_list:Job
        """
        queue = deque()
        for x in range(0, len(job_list)):
            queue.append(job_list[x])
        return queue

    def schedule(self, job_list):
        """
        Start to execute SJF scheduling algorithm on a job list sorted by arrival times.
        :param job_list:Job
        :return completed_jobs:Job
        """
        self.completed_jobs = []

        # sort all jobs based on the job arrival Time
        sorted_job_list = self.sort_job(job_list)

        # execute all jobs and get the completion time for each job
        self.run(sorted_job_list)

        return self.completed_jobs


if __name__ == '__main__':
    # Test 1
    job1 = Job(1, execution_duration=2, arrival_time=0)
    job2 = Job(2, execution_duration=3, arrival_time=1)
    job3 = Job(3, execution_duration=1, arrival_time=2)
    job4 = Job(4, execution_duration=2, arrival_time=3)

    scheduler = SJF()
    completed_jobs = scheduler.schedule([job1, job2, job3, job4])
    for job in completed_jobs:
        print("JobId:", job.get_job_id(),
              "Execution time:", job.get_execution_duration(),
              "Arrival Time:", job.get_arrival_time())
        print("Completion time:", job.get_completion_time())
        print("Waiting time:", job.get_waiting_duration())
        print("-----------------------------------------------------------------")

    # Test 2
    job1 = Job(1, execution_duration=5, arrival_time=2)
    job2 = Job(2, execution_duration=3, arrival_time=2)

    scheduler = SJF()
    completed_jobs = scheduler.schedule([job1, job2])
    for job in completed_jobs:
        print("JobId:", job.get_job_id(),
              "Execution time:", job.get_execution_duration(),
              "Arrival Time:", job.get_arrival_time())
        print("Completion time:", job.get_completion_time())
        print("Waiting time:", job.get_waiting_duration())
        print("-----------------------------------------------------------------")
