import os

import numpy as np

from constants import PATH, NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS, SEED, pkl_name, boris
from data_preprocessor import read_rating
from mf import MF
from serializer import dump, load

np.random.seed(SEED)


def run(my_params):
    filename, results_name = pkl_name(my_params)

    if os.path.exists(filename):
        model = load(filename)
    else:
        rating = read_rating(PATH, NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS)
        model = MF(
            rating.R,
            K=my_params.latent_factor,
            alpha=my_params.lr,
            beta=my_params.reg,
        )

    split = 5
    for i in range(int(my_params.epoch / split)):
        # dump each 5 iterations
        model.train(split)
        print(model.training_process)
        dump(model, filename)
        dump(model.training_process, results_name)
        print("dumped to {}".format(filename))
        print("dumped results to {}".format(results_name))

        if model.iter_done >= my_params.epoch:
            print("done {} iterations, exiting".format(model.iter_done))
            break



for p in boris:
    run(p)
