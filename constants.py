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


MFParams = namedtuple('MFParams', [
    'lr',
    'latent_factor',
    'reg',
    'epoch'
])

params = MFParams(0.001, 5, 0.001, 20)

# Anna
params1 = MFParams(0.001, 20, 0.001, 200)
params2 = MFParams(0.01, 20, 0.001, 200)
params3 = MFParams(0.01, 10, 0.001, 200)
params4 = MFParams(0.001, 10, 0.001, 200)

anna = [params1, params2, params3, params4]
# Boris

params5 = MFParams(0.001, 5, 0.001, 1000)
params6 = MFParams(0.01, 5, 0.001, 1000)
params7 = MFParams(0.001, 50, 0.001, 1000)

boris = [params5, params6, params7]


def pkl_name(mf_params: MFParams) -> (str, str):
    return ("pickle/mf_{}_{}_{}.pkl".format(mf_params.lr, mf_params.latent_factor, mf_params.reg),
            "pickle_res/mf_{}_{}_{}.pkl".format(mf_params.lr, mf_params.latent_factor, mf_params.reg))
