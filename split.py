import argparse
import pickle
import os
import random

data = 'citeseer'

with open("HNHN_data/%s/features.pickle" % (data), 'rb') as file:
    features = pickle.load(file)
with open("HNHN_data/%s/labels.pickle" % (data), 'rb') as file:
    labels = pickle.load(file)

print(len(labels))

if not os.path.exists(f"HNHN_data/{data}/splits/splits_16"):
    os.makedirs(f"HNHN_data/{data}/splits/splits_16")

for i in range(1,11):
    # make a new split that is 8% train and 92% test
    numbers = list(range(len(labels)))

    random.shuffle(numbers)
    train = numbers[:int(len(numbers)*0.16)]
    test = numbers[int(len(numbers)*0.16):]

    new_split = {'train': train, 'test': test}

    # dump to pickle
    with open(f"HNHN_data/{data}/splits/splits_16/{i}.pickle", 'wb') as file:
        pickle.dump(new_split, file)