import pickle


def dump(model, file_name):
    # open a file, where you ant to store the data
    file = open(file_name, 'wb')

    # dump information to that file
    pickle.dump(model, file)

    # close the file
    file.close()


def load(file_name):
    # open a file, where you stored the pickled data
    file = open(file_name, 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    return data
