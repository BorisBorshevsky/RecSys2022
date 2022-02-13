import concurrent.futures
import os

import numpy as np

from constants import SEED, pkl_name, boris, anna
from data_preprocessor import read_ratings
from mf import MF
from serializer import dump, load

np.random.seed(SEED)


def run(my_params, data_set='1m'):
    print("running on params: {}".format(my_params))
    filename, results_name = pkl_name('mf', data_set, my_params)

    if os.path.exists(filename):
        print("start loading {}".format(filename))
        model = load(filename)
        print("done  loading {}".format(filename))
    else:
        rating = read_ratings(data_set)
        model = MF(
            rating.R,
            K=my_params.k,
            alpha=my_params.lr,
            beta=my_params.reg,
        )

    split = 5
    for i in range(int(my_params.epoch / split)):
        # dump each {split} iterations
        print("training: {} for {} iter, done:{}/{}".format(params, split, model.iter_done, my_params.epoch))
        model.train(split)
        print(model.training_process)
        dump(model, filename)
        dump(model.training_process, results_name)
        print("dumped to {}".format(filename))
        print("dumped results to {}".format(results_name))

        if model.iter_done >= my_params.epoch:
            break

    print("done {} iterations, exiting".format(model.iter_done))
    return model


def safe_run(param, data_set):
    try:
        run(param, data_set)
    except Exception as e:
        print(param, data_set, e)


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures_dict = {executor.submit(safe_run, p, '100k'): p for p in boris + anna}
    for future in concurrent.futures.as_completed(futures_dict):
        params = futures_dict[future]
        try:
            model = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (params, exc))
        else:
            print("done params: {}".format(params))
