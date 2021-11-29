


import dill as pickle


"""
==================================================================================================================
Save and load data from the model
==================================================================================================================
"""

def save_data(filename, model, other=[]): ### filename = full/relative path w/o extension, other- list
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'

    with open(filename, 'wb') as filehandler:
        data = list(pick_model_data(model)) + other
        pickle.dump(data, filehandler)

    if model.flags['verbose']:
        print("Object data is saved to {0:s}".format(filename))


def load_data(filename):
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'

    with open(filename, 'rb') as filehandler:
        data = pickle.load(filehandler)

    return data


# def load_data(filename, model):
#     if not filename.endswith('.pkl'):
#         filename = filename + '.pkl'

#     with open(filename, 'rb') as filehandler:
#         data = pickle.load(filehandler)

#     for i, d in enumerate(pick_model_data(model)):
#         d[:] = data[i]

#     other = data[i+1:]
#     return other


def pick_model_data(model):
    yield model.observations
    yield model.Energy_elastic
    yield model.Energy_kinetic