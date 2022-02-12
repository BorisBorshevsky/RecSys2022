import os

import numpy as np

from constants import PATH, NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS, SEED, params, pkl_name
from data_preprocessor import read_rating
from mf import MF
from serializer import dump, load

np.random.seed(SEED)

filename = pkl_name(params)

if os.path.exists(filename):
    model = load(filename)
else:
    rating = read_rating(PATH, NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS)
    model = MF(
        rating.R,
        K=params.latent_factor,
        alpha=params.lr,
        beta=params.reg,
    )

for i in range(100):
    # dump each 5 iterations
    model.train(5)
    print(model.training_process)
    dump(model, filename)
    print("dumped to {}".format(filename))
