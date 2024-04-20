from job import Job
from collections import deque

from util import *


class FCFSWithDyanmicBatching:
    """
    This class is used to calculate the waiting time and total completion time of each job.
    The scheduling algorithm is first-come-first-serve where every time the job that arrives the earliest in the queue
    will be scheduled to start executing.

    The scheduler supports dynamic batching and the batch size is determined by the MAX_BATCH_SIZE parameter.
    The scheduler will try to batch incoming jobs into batches with wait timeout of BATCH_WAIT_TIMEOUT. The two parameters
    are defined in the util.py file.
    Dynamic batching is also called continuous batching. Re-batching is performed when at least 1 job in the batch has finished.
    """
    queue = deque()
    completed_jobs = []
    completion_time = 0

    @staticmethod
    def sort_job(job_list):
        """Sort the job list based on arrival time.
        :param job_list:Job
        :return sorted_job_list:Job
        """
        sorted_job_list = sorted(job_list, key=lambda x: x.arrival_time, reverse=False)
        return sorted_job_list

    def run(self, job_list):
        """Run FCFS algorithm on the job list.
        Calculate the completion time of each job and stores calculated completion_time in the job object.
        Assume that the job list has been sorted based on the arrival times.
        :param job_list:Job
        """

        self.convert_to_queue(job_list)

        self.completion_time = 0
        time_slice = 1
        while len(self.queue) > 0:
            running_batch = []
            if self.completion_time < self.queue[0].get_arrival_time():
                self.completion_time = self.queue[0].get_arrival_time()
            while len(self.queue) > 0 and len(running_batch) < MAX_BATCH_SIZE and self.queue[0].get_arrival_time() <= self.completion_time + BATCH_WAIT_TIMEOUT:
                running_batch.append(self.queue.popleft())
            # get the job_arrival_time and job_duration based on the running_batch
            if len(running_batch) == MAX_BATCH_SIZE:
                job_arrival_time = running_batch[-1].get_arrival_time()
            else:
                job_arrival_time = self.completion_time + BATCH_WAIT_TIMEOUT
            # job_duration = max([x.get_execution_duration() for x in running_batch])  # the longest job duration in the batch
            job_duration = min([x.get_execution_duration() for x in running_batch])  # the shortest job duration in the batch

            # wait until the arrival time has been reached
            if round(time_slice, 2) == 0:
                time_slice = 1
            while self.completion_time < job_arrival_time:
                if self.completion_time + time_slice < job_arrival_time:
                    self.completion_time += time_slice
                else:
                    remaining_slice = 1 - (job_arrival_time - self.completion_time)
                    self.completion_time = job_arrival_time
                    time_slice = remaining_slice

            # enter the running stage
            for running in running_batch:
                running.set_status("RUNNING")
                running.set_waiting_duration(self.completion_time - running.get_arrival_time())

            job_duration_to_run = job_duration
            while job_duration > 0:
                job_duration -= time_slice
                if job_duration < 0:
                    self.completion_time += job_duration + time_slice
                    remaining_slice = abs(job_duration)
                    time_slice = remaining_slice
                else:
                    self.completion_time += time_slice
                    time_slice = 1

            # job completed
            for running in running_batch:
                if running.get_execution_duration() <= job_duration_to_run:
                    running.set_status("COMPLETED")
                    running.set_completion_time(self.completion_time)
                    self.completed_jobs.append(running)
                else:
                    # re-batching by inserting the unfinished job back to the queue
                    running.set_status("WAITING")
                    running.set_execution_duration(max(running.get_execution_duration() - job_duration_to_run, 0))
                    self.queue.appendleft(running)
        print("Completion time after executing all jobs:", self.completion_time)

    def convert_to_queue(self, job_list):
        """Convert the given job list to queue.
        :param job_list:Job
        """
        for x in range(0, len(job_list)):
            self.queue.append(job_list[x])

    def schedule(self, job_list):
        """
        Start to execute FCFS scheduling algorithm on a job list sorted by arrival times.
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
    job2 = Job(2, execution_duration=2, arrival_time=1)
    job3 = Job(3, execution_duration=2, arrival_time=2)
    job4 = Job(4, execution_duration=2, arrival_time=3)

    scheduler = FCFSWithDyanmicBatching()
    completed_jobs = scheduler.schedule([job1, job2, job3, job4])
    for job in completed_jobs:
        print("JobId:", job.get_job_id(),
              "Execution time:", job.get_execution_duration(),
              "Arrival Time:", job.get_arrival_time())
        print("Completion time:", job.get_completion_time())
        print("Waiting time:", job.get_waiting_duration())
        print("-----------------------------------------------------------------")

    # Test 2
    job1 = Job(1, execution_duration=2, arrival_time=2)
    job2 = Job(2, execution_duration=3, arrival_time=2)

    scheduler = FCFSWithDyanmicBatching()
    completed_jobs = scheduler.schedule([job1, job2])
    for job in completed_jobs:
        print("JobId:", job.get_job_id(),
              "Execution time:", job.get_execution_duration(),
              "Arrival Time:", job.get_arrival_time())
        print("Completion time:", job.get_completion_time())
        print("Waiting time:", job.get_waiting_duration())
        print("-----------------------------------------------------------------")
