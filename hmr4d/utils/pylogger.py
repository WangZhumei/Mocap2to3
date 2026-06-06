from time import time


class Log:
    log_on = True

    @staticmethod
    def info(*args):
        if Log.log_on:
            print("\033[1;32m[INFO]\033[0;0m", *args)

    @staticmethod
    def warn(*args):
        if Log.log_on:
            print("\033[1;35m[WARN]\033[0;0m", *args)

    @staticmethod
    def error(*args):
        if Log.log_on:
            print("\033[1;31m[ERROR]\033[0;0m", *args)

    @staticmethod
    def debug(*args):
        if Log.log_on:
            print("\033[1;33m[DEBUG]\033[0;0m", *args)

    def time():
        return time()


def monitor_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""

    def wrapped(*args, **kwargs):
        Log.info(f'"{func.__name__}()" begin...')
        ret_value = func(*args, **kwargs)
        Log.info(f'"{func.__name__}()" end...')
        return ret_value

    return wrapped


def monitor_class_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""

    def wrapped(self, *args, **kwargs):
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" begin...')
        ret_value = func(self, *args, **kwargs)
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" end...')
        return ret_value

    return wrapped
