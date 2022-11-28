import numpy as np

ground_truth = []
predictions = []

with open('ground_truth.txt') as f:
    for line in f:
        ground_truth.append(int(line.strip()))

with open('predictions.txt') as f:
    for line in f:
        predictions.append(int(line.strip()))

acc = np.mean(np.array(ground_truth) == np.array(predictions))
print(acc)