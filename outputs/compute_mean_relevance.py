import json

with open("outputs/guwenbert_llm_judge_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["details"]

total_score = 0
total_chunks = 0

for q in results:
    for chunk in q["judged_chunks"]:
        total_score += chunk["judge_score"]
        total_chunks += 1

mean_relevance = total_score / total_chunks if total_chunks > 0 else 0

print("Mean Relevance:", round(mean_relevance, 4))