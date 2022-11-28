
expert_results = []
predictions = []

with open('CCPM/expert_results.txt', 'r') as f:
    for line in f:
        expert_results.append(int(line.strip()))

with open('predictions.txt', 'r') as f:
    for line in f:
        predictions.append(int(line.strip()))

for i in range(20):
    if expert_results[i] != predictions[i]:
        print(i+1)