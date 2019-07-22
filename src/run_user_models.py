from stock_backtest import ddpg_restore_model
from pymongo import MongoClient
import datetime
import quandl
import traceback
import logging


def run_model(train_id, start_date, end_date):
    model = ddpg_restore_model(train_id)
    model.restore()
    summary = model.backtest(start_date, end_date)
    return summary


def to_iso_date(date):
    return str(date)[:10]


def get_model(model_id):
    model = db["models"].find_one({"model_id": model_id})
    return model


def get_train_id(model_id):
    model = get_model(model_id)
    return model["train_id"]


if __name__ == '__main__':
    logger = logging.getLogger("run_user_models")

    quandl.ApiConfig.api_key = "AMx5b_Ww3czWcfcKNaA4"

    client = MongoClient()
    db = client['deepfolio']

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    end_date = to_iso_date(tomorrow)

    # Start back testing models that in user's running model list
    # For every model that users has selected to run in real market,
    # we do back-testing from the running start date to tomorrow
    # The last day weight change will be taken as latest trading signal
    users = db["user_models"].find()
    for user in users:
        for user_model in user["models"]:
            try:
                model_id = user_model["model_id"]
                train_id = get_train_id(model_id)
                start_date = user_model["run_date"]
                if start_date is not None:
                    summary = run_model(train_id, start_date, end_date)
                    user_model["summary"] = summary
            except Exception as e:
                logging.error(traceback.format_exc())

        db["user_models"].save(user)
