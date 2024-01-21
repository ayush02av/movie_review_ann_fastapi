import pickle as pkl

with open('model.pkl', 'rb') as file:
    print(pkl.load(file))