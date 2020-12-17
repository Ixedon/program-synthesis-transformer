from concurrent import futures
import time


def inf(name):
    a = 0
    while True:
        a += 1
        print(a, f"Name:{name}")
        time.sleep(a)


executor = futures.ThreadPoolExecutor(max_workers=1)

for i in range(1000):
    future = executor.submit(inf, str(i))
    try:
        compiled, passed = future.result(10)
        #             print(program)
        executor._threads.clear()
        futures.thread._threads_queues.clear()
    #             return program, args
    except futures.TimeoutError:
        print("Timeout")

        executor._threads.clear()
        futures.thread._threads_queues.clear()
