import os

import matplotlib.pyplot as plt

from constants import params, pkl_name, MFParams
from serializer import load

init_params = params


def mf_data(params):
    data, results = pkl_name(params)
    model = load(data)
    return model.training_process, "RMSE - lr {} lf {}".format(params.lr, params.latent_factor)


def draw_plot():
    models = os.listdir('pickle_res')
    for model in models:
        filename = model.replace(".pkl", "")
        alg, lr, lf, reg = filename.split("_")
        mf_params = MFParams(float(lr), int(lf), float(reg), 0)
        res_init_params, label = mf_data(mf_params)

        y = [r[1] for r in res_init_params]
        x = [r[0] for r in res_init_params]
        plt.plot(x, y, label=label)

    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('iter')

    plt.show()


draw_plot()
