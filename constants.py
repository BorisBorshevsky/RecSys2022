from collections import namedtuple

import time

DATA_FILE_NAME = 'ml-1m'
NUM_USERS = 6040
NUM_ITEMS = 3952
NUM_TOTAL_RATINGS = 1000209
TRAIN_RATIO = 0.9
PATH = "./data/%s" % DATA_FILE_NAME + "/"
SEED = 1234


def get_results_path(optimizer_method, lr):
    return './results/{DATA_FILE_NAME}/{SEED}_{optimizer_method}_{lr}_{time}/'.format(DATA_FILE_NAME=DATA_FILE_NAME,
                                                                                      SEED=SEED,
                                                                                      optimizer_method=optimizer_method,
                                                                                      lr=lr,
                                                                                      time=str(time.time()).split(".")[
                                                                                          0])



RunParams = namedtuple('MFParams', [
    'lr',
    'k',
    'reg',
    'epoch'
])

params = RunParams(0.001, 5, 0.001, 20)

iters = 2000

# Anna
params1 = RunParams(0.001, 50, 0.001, iters)
params2 = RunParams(0.005, 20, 0.001, iters)
params3 = RunParams(0.005, 10, 0.001, iters)
params4 = RunParams(0.001, 10, 0.001, iters)

anna = [params1, params2, params3, params4]
# Boris

params5 = RunParams(0.001, 5, 0.001, iters)
params6 = RunParams(0.01, 5, 0.001, iters)
params7 = RunParams(0.005, 50, 0.001, iters)

boris = [params5, params6, params7]


def pkl_name(alg: str, mf_params: RunParams) -> (str, str):
    return ("pickle/{}_{}_{}_{}.pkl".format(alg, mf_params.lr, mf_params.k, mf_params.reg),
            "pickle_res/{}_{}_{}_{}.pkl".format(alg, mf_params.lr, mf_params.k, mf_params.reg))
