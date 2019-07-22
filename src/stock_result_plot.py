import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
from utils.modelutils import get_model_path, get_result_path, load_config, build_parser, plotPV, plotWeight, max_drawdown, alpha_value, sharpe_ratio, change_in_value


if __name__ == '__main__':
    train_id_list = []
    #read all the files in train_package directory 
    path = "C:/yintech/model/src/train_package"
    for filename in os.listdir(path):
        train_id_list.append(filename)

    # train_id_list = ["HMC","IBKR","NTES","OSN","RCON","SDRL","SINA"]
    
    pv_list = []
    average_return_list = []
    pv = []
    average_return = []
    date_list = []
    ticker = ""

    for train_id in train_id_list:
        result_path = get_result_path(train_id) + "summary.json"
        # print(result_path)
        with open(result_path,'r') as outfile:
            f = json.load(outfile)
        print(train_id)
        ticker = train_id
        pv = f['pv']
        average_return = f['average_return']
        v = pv[-1]
        a = average_return[-1]
        print("the pv of the portfolio is:{}".format(v))
        print("the market value of the portfolio is:{}".format(a))
        pv_list.append(v)
        average_return_list.append(a)
        max_draw = max_drawdown(pv)
        date_list = f['date']
        alpha = alpha_value(pv,average_return)
        sharpe = sharpe_ratio(pv,average_return)
        print("max drawdown is: {}".format(max_draw))
        print("alpha_value is: {}".format(alpha))
        print("sharpe ratio is: {}".format(sharpe))
        

        df = pd.DataFrame({"date" : date_list, ticker : pv , "market_value":average_return})
        df['date'] = pd.to_datetime(df['date'])  
        print(df.head())     
        df.plot(x = 'date', y = [ticker, 'market_value'], legend = True, title = "AI agent performance")
        plt.show()

    # print("length of weight_forchange:{}".format(len(weight_forchange)))
    
    print(pv_list)
    print(average_return_list)

    
    
