Copyright 2018, Yintech Innovation Labs, All rights reserved.

Written by Hanxiong Wang.

# DDPG model code introducetion

In the source code, the mainly components are two classes: 
* ddpg_trading_train in stock_model_train.py
* ddpg_restore_model in stock_test.py

And the configuration file is `stock.json` under `config/` folder

## Configuration
We give the instruction for setting the parameters in configuration file.
* input
  * "predictor_type": we provide two kinds of networks: `"lstm"` and `"cnn"`. The default setting is `"lstm"`
  * "stocks": a list that provides the stocks portfolios
  * "windows_length": the number of time windows as input data
  * "use_batch_norm": the default setting is `true`, this is used to come over vanishing gradient problem. If we set it as `false`, ideally speaking, we can speed up the training procedure.
  * "training_start_time": `"YYYY-MM-DD"`,
  * "training_end_time": `"YYYY-MM-DD"`, training end time shall always be later then training start time
  * "eval_start_time": `"YYYY-MM-DD"`,
  * "eval_end_time": `"YYYY-MM-DD"`. In this stage, it is not useful. We shall simply set them as the same as `testing_start_time` and `testing_end_time`
  * "trading_cost": the default setting is `0`,
  * "time_cost": the default setting is `0`,
  * "feature_number": the default setting is `5`,
    * 1 means the feature is ["close/open"],
    * 2 means the features are ["close/open", "volume"],
    * 4 means the features are ["open", "low", "high", "close"]
    * 5 means the features are ["open", "low", "high", "close", "volume"]
    * We shall notice for 4 and 5, "open", "low", "high" and "close" are relatively values and "volume" are normalized as z-scores
  * "reward_function": the default setting is `2`,
    * 1 means the origin paper reward function
    * 2 means our reward function followed by the bitcoins paper
* layers
  * activations_function: the default setting is `"prelu"`. Could be "tanh","relu","leaky_relu" and other activation functions
* training
  * "gamma": the default setting is `0.99`. It is the reward decay coefficient. 0.99 is the common setting
  * "action_bound": `1`. This could not be larger then 1
  * "critic learning rate": `0.001`,
  * "actor learning rate": `0.001`,
  * "critic decay rate": `0.99`,
  * "actor learning rate": `0.99`,
  * "critic decay steps": `400`,
  * "actor decay steps": `400`,
    * the above "actor" and "critic" setting are for the learning rate and decay speed. low learning rate means finer training, but maybe much lower speed to reach the optimal solution. The default setting here are suitable for the episode 200. We can change the learning rate and decay rate depend on the different episodes.
  * "critic weight decay": `0.06`,
  * "actor weight decay": `0.06`,
    * the above two parameters are used for "L2 regularization". Could speed up the training and avoid overfitting. But large number of "L2 regularization" will lower the ability of the model
  * "tau": `0.001`, soft update for target network
  * "batch size": `64`. It can be larger as 128 or 256.
  * "max_step": default as `0`. if it is 0, that means in each episode, we experience all the training　periods. Otherwise, it must be smaller then the total training days.
  * "seed": `1337`, you can set any number as you wish.
  * "buffer size": `20000`, the size of reply buffer. To some extent, it also depend on the total episodes.
  * "episode": `200`, the size of epsidoes
* testing
  * testing_start_time: `"YYYY-DD-MM"`,
  * testing_end_time: `"YYYY-DD-MM"`,  testing end time shall always be later then testing start time
  
## Working flow
1. First, modified the `stock.json` file under `config/stock.json` as needed
2. After loading `config/stock.json`, we call ddpg_trading_train('config') .train_model() to
initialize and train the model.
3. When the model complete training, it will generate a subfolder under `train_package` and name with `train_id`. Both model (with ckpt files) and tensorboard files will be stored under this subfolder.
4. We call ddog_restore_model(train_id).restore() to restore the model.
5. We can use the ddpg_restore_model.backtest() to make the backtesting.
6. The backtesing result will be saved in the `summary.json`

## Miscellaneous
* `main.py` can be directly run by `python main.py`, generate the model and complete the backtesing according to the config file.
* All the model will be saved by naming with the number order starting from 0. So if you generate a model named with `n` but fail to complete the training, you shall delete the folder `n`. Otherwise, you will have the folder `n+1` to store your next training model
* If we use `find_best_model.py`, simply run `python find_best_model.py` we will train and do the backtest for the same stock portfolios several times and pick the best model. The best model will be named with stocks name instead of number anymore. The other worse models will be removed.

## Platform Support
Python 3.5+ in windows and Python 3.5+ in linux are supported.

## Dependencies
Install Dependencies via `pip install -r requirements.txt`.

Personally, I suggested use [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) to manage your python environment.

# AWS auto deploy tool Introduction
This tool aims to automatically launch the AWS EC2 instances as the numbers we wished from the prepared ami image. The launched instances, or we can call it "AI worker", will automatically fetch the tasks from the specified bucket on S3 and run the task.

* For each AI worker, it will always run one and only one task. When the task is finished, the AI worker will first upload the results as a zip file to the specified bucket on S3, then fetch another task by reading uncompleted tasks list on S3.

* This tool is under the `aws` folder. Actually, it is totally independent with other parts in the `src` folder although we put the code under `src` folder in this stage.

## How to start instances from the console end
* We will use `boto3` package to communicate with AWS. To install `boto3`, we simply run [`pip install boto3`](https://boto3.readthedocs.io/en/latest/guide/quickstart.html)

* Before you can begin using Boto 3, you should set up authentication credentials. Credentials for your AWS account can be found in the IAM Console. You can create or use an existing user. https://boto3.readthedocs.io/en/latest/guide/quickstart.html

* In the console end, we run `main.py` to create instances on EC2 and let AI worker start to work instantly.
    * Before we run the`main.py`, we need find out the correct ami image. For example, we can use image name with `auto_deploy_Aug141032`, which image id is `ami-0a448b620338476b0` in this stage. Then the proper command is `python main.py -i=ami-ami-0a448b620338476b0`
    * We can set up the `instances_number` and correct `bucket_name` in S3 in the source code of `main.py`. Also, the type of instance can be manually given in the `create_ec2_instances` class.
    * After the instances are created and running successfully, a json file called `instances_info.json` would be created under directory `aws/tmp/`. The instance id and named is recorded in that file.
    * When all the tasks are completed, we can run `python terminate_instance.py` to close all the instances we created and the `instances_info.json` will be removed simultaneously.

## Configuration file of tasks
* Each time before we try to run the tasks, we need generate a json file which contain the basic parameters for the DDPG model, the portfolios list and their corresponding test date list. Here, we put a `tasksconfig.json` under directory `aws/config` as an example. You can generate a similar json file with a much larger portfolios list for future use.

* This `tasksconfig.json` file will be upload to S3 and the AI workers will fetch the tasks from this file.

## Working flow at EC2 end
* When we launch the instances in EC2, we simultaneously run the script by user_data. Actually, we mainly run `aws_task_run.py` under folder `src`. This python file do the following work:
    * First, it try to download the `tasksconfig.json` from S3. If it fails, it will wait for 2 seconds and retry. Otherwise, it will download the file and delete it in S3 bucket. The result will record in the `/tmp/download_log.txt`
    * Loading and fetch the task, it will upload the modified `tasksconfig.json` to S3 with remaining task.
    * `aws_test_run.py` will be called to compute the task. The result will be compressed as a zip file and upload to S3 bucket
    * When there is no more tasks remaining in the `tasksconfig.json`, it's done.
    * Notice `/tmp/working_log.txt` will record the tasks computed by each AI worker. And the `/tmp/error_log.txt` will record the portfolio when it is accessed by two AI workers.

## Miscellaneous
* In this stage, we are running commands on linux instance at launch by user_data. We can also access the control the instance by ssh method. I keep this kind of option in `main.py` in the comment part.

* In this stage, we have two versions when creating instances. 
    * In the branch "hanxiong", the ami-image I used here is `ami-0a448b620338476b0`, the environment variables in this image contains `aws_access_key_id` and `aws_secret_access_key`, so there maybe some security issues.
    * In the branch "aws_auto_role", the ami-image I used here is `ami-0007139a1e642fcc0`. `aws_access_key_id` and `aws_secret_access_key` are not in the environment variables. And we created the instances with IAM role. So it is more safe. But in the beginning, the code does not work. I have no idea how I fix the problem. 
    * In addition, when you need to use version with IAM role, remember to set the enough `Maximum CLI/API session duration` on aws before launching the instances
