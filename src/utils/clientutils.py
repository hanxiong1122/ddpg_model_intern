import json
import os
import sys

import numpy as np
import redis
from pymongo import MongoClient


class RedisQueue(object):
    _dict = {}

    """Simple Queue with Redis Backend"""
    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.key = '%s:%s' %(namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def clear(self):
        """Clear the queue of all messages, deleting the Redis key."""
        self.__db.delete(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)

        if item:
            item = item[1]
        return item

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

    @classmethod
    def get_queue(cls, qname):
        return cls._dict.setdefault(qname, RedisQueue(qname))




class Stage:
    START_TRAINING = "StartTraining"
    TRAINING = 'Training'
    START_TESTING = 'StartTesting'
    TESTING  = 'Testing'
    DONE     = 'Done'
    ERROR    = 'Error'


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def getRedis():
    redis_host = 'localhost'
    if 'REDIS_HOST' in os.environ:
        redis_host = os.environ['REDIS_HOST']
    r = redis.StrictRedis(host=redis_host, port=6379, db=0, decode_responses=True)
    return r

r = getRedis()

mongo_client = MongoClient()


def need_report_progress():
    return "-r" in sys.argv


def need_save_to_db():
    return "-db" in sys.argv


def report_model_progress(model_id, info):
    if need_report_progress():
        info["model_id"] = model_id
        if r is not None:
            r.publish("progress", json.dumps(info, cls=MyEncoder))


def save_test_result_to_db(result):
    if need_save_to_db():
        db = mongo_client['deepfolio']
        db["models"].insert_one(result.copy())

