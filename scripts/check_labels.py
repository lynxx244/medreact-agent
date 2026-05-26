import json
data = json.load(open('results/eval_medreact_v8.json', encoding='utf-8'))
wrong = [r for r in data['results'] if r['ground_truth_risk']=='中' and r['predicted_risk']=='高']
print(f'中风险被判高风险: {len(wrong)}条')
for i, r in enumerate(wrong):
    print(f'{i+1}. {r["symptoms"]}')
    print()
