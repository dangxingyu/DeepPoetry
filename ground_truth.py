import jsonlines

results = []
with jsonlines.open('CCPM/valid.jsonl') as reader:
    for obj in reader:
        # print(obj)
        results.append(obj['answer'])

with open('ground_truth.txt', 'w') as f:
    for i in results:
        f.write(f'{i}\n')