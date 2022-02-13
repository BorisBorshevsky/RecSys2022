import os

import matplotlib.pyplot as plt

from constants import pkl_name, RunParams
from serializer import load


def data_load(alg: str,data_set:str, params: RunParams):
    data, results = pkl_name(alg,data_set, params)
    model = load(results)
    return model, "RMSE - {} - lr={} k={} l={}".format(alg, params.lr, params.k, params.reg)


algs = frozenset(['mf', 'Adam-AutoRec'])


def draw_plots(algs=algs,data_set='1m', limit=100):
    models = os.listdir('pickle_res')
    for model in models:
        filename = model.replace(".pkl", "")
        alg, lr, lf, reg = filename.split("_")
        if alg in algs:
            params = RunParams(float(lr), int(lf), float(reg), 0)
            res_init_params, label = data_load(alg,data_set, params)

            y = [r[1] for r in res_init_params]
            x = [r[0] for r in res_init_params]
            plt.plot(x[:limit], y[:limit], label=label)

    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')

    plt.show()

if __name__ == '__main__':
    draw_plots(algs=frozenset({'Adam-AutoRec'}), limit=200)
    # draw_plots(algs=frozenset({'mf'}), limit=200)
    # draw_plots(limit=200)
