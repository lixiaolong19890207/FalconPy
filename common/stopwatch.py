# -*- coding:utf-8 -*-
"""
@File    : stopwatch.py
@Time    : 2018/10/14
"""
import os
import time

from common.log import logger


def stopwatch(file_name):
    file_name = os.path.basename(file_name)

    def decorator(func):
        def inner(*args, **kwargs):
            start_t = time.time()
            logger.info('StopWatch(): %s-%s() begin...' % (file_name, func.__name__))

            ret = func(*args, **kwargs)

            logger.info('~StopWatch(): %s-%s() need time: %fs' % (file_name, func.__name__, time.time() - start_t))
            return ret

        return inner

    return decorator


if __name__ == '__main__':
    pass
