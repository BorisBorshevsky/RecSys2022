import os

import matplotlib.pyplot as plt

from constants import pkl_name, RunParams
from serializer import load


def mf_data(params):
    data, results = pkl_name('mf', params)
    model = load(results)
    return model, "RMSE - lr={} k={}".format(params.lr, params.latent_factor)

def autorec_data(params):
    data, results = pkl_name('Adam-AutoRec', params)
    model = load(results)
    return model, "RMSE - Adam-AutoRec - lr={} k={}".format(params.lr, params.latent_factor)



def draw_plot_mf(limit=100):
    models = os.listdir('pickle_res')
    for model in models:
        filename = model.replace(".pkl", "")
        alg, lr, lf, reg = filename.split("_")
        if alg == "mf":
            mf_params = RunParams(float(lr), int(lf), float(reg), 0)
            res_init_params, label = autorec_data(mf_params)

            y = [r[1] for r in res_init_params]
            x = [r[0] for r in res_init_params]
            plt.plot(x[:limit], y[:limit], label=label)

    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')

    plt.show()

def draw_plot_autorec(limit=100):
    models = os.listdir('pickle_res')
    for model in models:
        filename = model.replace(".pkl", "")
        alg, lr, lf, reg = filename.split("_")
        if alg == "Adam-AutoRec":
            mf_params = RunParams(float(lr), int(lf), float(reg), 0)
            res_init_params, label = autorec_data(mf_params)

            y = [r[1] for r in res_init_params]
            x = [r[0] for r in res_init_params]
            plt.plot(x[:limit], y[:limit], label=label)

    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')

    plt.show()


if __name__ == '__main__':
    # draw_plot_mf(200)
    draw_plot_autorec(200)
