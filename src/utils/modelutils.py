import json
import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_train_package_dir():
    file_dir= os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(file_dir, "../train_package")
    return path


def get_config_path(train_id):
    return get_train_package_dir() + "/" + str(train_id) + "/stock.json"


def get_model_path(train_id):
    return 'train_package/{}/weights/checkpoint.ckpt'.format(train_id)


def get_result_path(train_id):
    return 'train_package/{}/results/'.format(train_id)


def generate_train_folder(config):
    train_id = 0
    while True:
        model_save_path = get_model_path(train_id)
        summary_path = get_result_path(train_id)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
            os.makedirs(summary_path, exist_ok=True)
            with open("./train_package/" + str(train_id) + "/stock.json", 'w') as outfile:
                json.dump(config, outfile, indent=4, sort_keys=True)
            logging.debug("creat new model in %s" % train_id)
            break
        else:
            train_id += 1
    return model_save_path, summary_path, train_id


def load_config(config_path="./config/stock.json"):
    try:
        with open(config_path) as f:
            config = json.load(f)
            logging.info("successfully load config file at " + config_path)
        return config
    except:
        print("error config file or path")
        raise

def load_aws_config(config_path="./config/aws_config.json"):
    try:
        with open(config_path) as f:
            config = json.load(f)
            logging.info("successfully load config file at " + config_path)
        return config
    except:
        print("error aws config file or path")
        raise

def plotWeight(omega_list):
    plt.figure()
    plt.title("Weights Distribution")
    Omegas = np.asarray(omega_list)
    x = list(range(Omegas.shape[0]))
    for i in range(Omegas.shape[1]):
        if i == 0:
            plt.bar(x, Omegas[:, i], label="Cash")
        else:
            plt.bar(x, Omegas[:, i], bottom=np.sum(Omegas[:, 0:i], axis=1), label=str(i))
    plt.legend(loc=[1, 0])
    plt.show()


def plotPV(figure_list, name):
    plt.figure()
    plt.title("PV figure")
    for i, fig_ele in enumerate(figure_list):
        fig = np.asarray(fig_ele)
        cord = list(range(fig.shape[0]))
        # print(cord)
        plt.plot(cord, fig, linewidth=1.0, linestyle='-', label=name[i])
    plt.legend(loc=[1, 0])
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')
    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=False)
    parser.add_argument('--window_length', '-w', help='observation window length', required=False)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=False)
    parser.add_argument('--plot_weight', '-pw', help='whether to plot weights', required=False)
    parser.add_argument('--plot_pv', '-v', help='whether to plot pv', required=False)
    parser.add_argument('--train_id', '-i', help='input the training id', required=False)
    return parser




def max_drawdown(vec):
    drawdown = 0.0
    max_seen = vec[0]
    for val in vec[1:]:
        max_seen = max(max_seen, val)
        drawdown = max(drawdown, 1 - val / max_seen)
    return drawdown


def alpha_value(portfolio_values, average_return):
    '''
    @: param portfolio_values: a list of portfolio values during a trading process
    @: param average_return:  a list of average weighted portflio values 
    '''
    return portfolio_values[-1] - average_return[-1]


def sharpe_ratio(portfolio_values, average_return):
    # there is no std value if only one in the list
    if len(portfolio_values) <= 1:
        return 0
    return (portfolio_values[-1] - average_return[-1]) / np.std(portfolio_values)


def change_in_value(value_list, intial_value):

    last_value = intial_value
    value_change = []
    for i, value in enumerate(value_list):
            change = value - last_value
            value_change.append(change)
            last_value = value
    return value_change



# def max_drawdown(portfolio_values):
#     '''
#     @: param portfolio_values: a list of portfolio values during a trading process
#     '''
#     drawdown_list = []
#     max_benefit = 0
#     for i in range(len(portfolio_values)):
#         if portfolio_values[i] > max_benefit:
#             max_benefit = portfolio_values[i]
#             drawdown_list.append(0.0)
#         else:
#             drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
#
#     return max(drawdown_list)
