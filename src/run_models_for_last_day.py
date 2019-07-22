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


if __name__ == '__main__':
    logger = logging.getLogger("run_all_models")

    quandl.ApiConfig.api_key = "AMx5b_Ww3czWcfcKNaA4"

    data = quandl.get("EOD/SPY", rows=1, sort_order="asc")
    dates = data.index.values
    last_day = to_iso_date(dates[0])

    client = MongoClient()
    db = client['deepfolio']

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    end_date = to_iso_date(tomorrow)
    start_date = last_day

    models = db["models"].find()
    results = []
    try:
        for model in models:
            model_id = model["model_id"]
            train_id = model["train_id"]

            summary = run_model(train_id, start_date, end_date)
            result = {"model_id": model_id, "stocks": model["stocks"], "start_date": start_date,
                      "end_date": end_date, "summary": summary}
            results.append(result)
    except Exception as e:
        logging.error(traceback.format_exc())

    today = datetime.datetime.today().strftime('%Y-%m-%d')

    db["daily_run"].replace_one({"date": today}, {"date": today, "results": results}, True)
