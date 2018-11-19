import time

PRINT_TIMER = False


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        end = time.clock()
        interval = end - self.start
        if PRINT_TIMER:
            print('{} took {:.5f} seconds'.format(self.name, interval))