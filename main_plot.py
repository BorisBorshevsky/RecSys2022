import os

import matplotlib.pyplot as plt

from constants import pkl_name, RunParams
from serializer import load


def data_load(alg: str, data_set: str, params: RunParams):
    data, results = pkl_name(alg, data_set, params)
    model = load(results)
    lable = "RMSE-{}-{} - lr={} k={}".format(alg,data_set, params.lr, params.k, params.reg) \
        if alg == 'mf' else\
        "RMSE-{}-{} - k={} lambda={}".format(alg,data_set, params.k, params.reg)
    return model, lable


algs = frozenset(['mf', 'Adam-AutoRec'])


def draw_plots(algs=set({}), data_set='1m', limit=100, minimum_itres=5, str_contains=None, str_exclude=None):
    models = os.listdir('pickle_res/{}'.format(data_set))
    min = 200
    min_params = None
    best_model = None

    for model in models:
        if model == 'old':
            continue
        filename = model.replace(".pkl", "")
        alg, lr, lf, reg = filename.split("_")
        print(filename, str_contains in filename, (str_contains and str_contains in filename) and not (str_exclude and str_exclude in filename))
        if alg in algs or ((str_contains and str_contains in filename) and not (str_exclude and str_exclude in filename)):
            params = RunParams(float(lr), int(lf), float(reg), 0)
            res_init_params, label = data_load(alg, data_set, params)


            y = [r[1] for r in res_init_params]
            x = [r[0] for r in res_init_params]
            if len(y) < minimum_itres:
                continue

            for value in y:
                if value < min:
                    min = value
                    min_params = params
                    best_model = filename

            plt.plot(x[:limit], y[:limit], label=label.replace("Adam-", ""))

    print("best params: {}, {}, {}".format(min, min_params, best_model))


    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')
    plt.title("Data set: {}, algorithms:{}".format(data_set, list(algs)))
    plt.show()

if __name__ == '__main__':
    # draw_plots(algs=frozenset({'Adam-AutoRec'}), data_set='1m', limit=200)
    # draw_plots(algs=frozenset({'New-AutoRec'}), data_set='1m', limit=200)
    # draw_plots(algs={'mf'}, data_set='1m', limit=200)
    draw_plots(data_set='100k', limit=200, str_contains="mf_")
    # draw_plots(algs=frozenset({}), data_set='100k', limit=400, str_contains="-f")
    # draw_plots(algs={}, data_set='1m', limit=400, str_contains="New-AutoRec-", str_exclude="")
    # draw_plots(algs={}, data_set='100k', limit=400, str_contains="fsigmoid-gselu", str_exclude="")
    # draw_plots(algs=frozenset({'mf'}), data_set='100k', limit=200)
    # draw_plots(algs=frozenset({'mf'}), limit=200)
    # draw_plots(limit=200)
