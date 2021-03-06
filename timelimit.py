import signal
from contextlib import contextmanager
import random
import time

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def inf(name):
    a = random.randint(0, 20)
    print(f"{name} Time: {a}")
    time.sleep(a)
    return "ok"

if __name__ == '__main__':
    for i in range(1000):
        print(f"I {i}")
        try:
            with time_limit(10):
                a = inf(str(i))
        except TimeoutException:
            a = "time"
            print(f"Timeout: {i}")
        print(a)