from utils.modelutils import  load_config
from stock_model_train import ddpg_trading_train
from stock_backtest import ddpg_restore_model

config = load_config()
print(config)
config["model_id"] = 0
train_id = ddpg_trading_train(config).train_model()
model = ddpg_restore_model(train_id)
model.restore()
model.backtest()
print("Finish work on %s"%(train_id))
