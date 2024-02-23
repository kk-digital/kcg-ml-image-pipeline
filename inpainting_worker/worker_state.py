
import queue
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

class WorkerState:
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.job_queue = queue.Queue()