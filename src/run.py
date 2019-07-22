import json
import redis
import threading
import time
from utils.modelutils import  load_config
from stock_model_train import ddpg_trading_train
from stock_backtest import ddpg_restore_model
from utils.clientutils import *

def worker(train_param):
    try:
        config = load_config()
        config["model_id"] = 0  # this is will be overridden when driven from UI
        print (train_param)

        if train_param is not None:
            config["model_id"] = int(train_param["model_id"])
            config["input"]["stocks"] = train_param["stocks"]

            if "feature_number" in train_param:
                config["input"]["feature_number"] = int(train_param["feature_number"])

            if "training_start_date" in train_param:
                config["input"]["training_start_time"] = train_param["training_start_date"]

            if "training_end_date" in train_param:
                config["input"]["training_end_time"] = train_param["training_end_date"]

            if "testing_start_date" in train_param:
                config["testing"]["testing_start_time"] = train_param["testing_start_date"]

            if "testing_end_date" in train_param:
                config["testing"]["testing_end_time"] = train_param["testing_end_date"]

            if "episode" in train_param:
                config["training"]["episode"] = int(train_param["episode"])

            if "trading_cost" in train_param:
                config["input"]["trading_cost"] = float(train_param["trading_cost"])

        train_id = ddpg_trading_train(config, DEBUG=False).train_model()
        model = ddpg_restore_model(train_id)
        model.restore()
        model.backtest()
        print("Finish work on %s", train_id)
    except Exception as e:
        print("Fatal error in training %s", train_param["stocks"])
        report_model_progress(config["model_id"],info={"error": True, "message": str(e)})



redis_host = 'localhost'
print(redis_host)
r = redis.StrictRedis(host=redis_host, port=6379, db=0, decode_responses=True)
p = r.pubsub()
p.subscribe('model-task')

for m in p.listen():
    if m['type'] == 'message':
        param = json.loads(m['data'])
        t = threading.Thread(target=worker, args=[param])
        t.start()