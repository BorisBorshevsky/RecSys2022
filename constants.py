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
                                                                                      time=str(time.time()).split(".")[0])
