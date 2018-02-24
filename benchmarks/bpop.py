from batch_shuffle.write import rd
from threading import Thread

N_MESSAGES = 100
KEY = "test"

class Worker(Thread):

    def __init__(self, redis_instance, name):
        super().__init__()
        self.rd = redis_instance
        self.name = name

    def run(self):
        while True:
            response = rd.blpop(KEY)
            message = response[1].decode('utf-8')
            print(f"Worker {self.name} received a message: {message}")

Worker(rd, "1").start()
Worker(rd, "2").start()

for i in range(N_MESSAGES):
    rd.rpush(KEY, str(i))
