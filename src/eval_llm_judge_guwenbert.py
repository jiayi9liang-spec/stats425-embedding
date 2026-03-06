from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from src.judge_llm_guwenbert import judge_chunk
from src.retrieve_numpy_guwenbert import retrieve


def load_jsonl(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dcg(scores):
    total = 0.0
    for i, s in enumerate(scores):
        total += s / math.log2(i + 2)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--emb_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ethanyt/guwenbert-base")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out_path", type=str, default="outputs/guwenbert_llm_judge_results.json")
    args = parser.parse_args()

    qa = load_jsonl(args.qa_path)
    all_rows = []

    hit_count = 0
    mrr_total = 0.0
    ndcg_total = 0.0
    precision_total = 0.0

    for idx, row in enumerate(qa, start=1):
        qid = row.get("id", f"q{idx}")
        question = row["question"]
        answer = row.get("answer", "")

        retrieved = retrieve(
            question=question,
            emb_path=args.emb_path,
            meta_path=args.meta_path,
            model_name=args.model_name,
            k=args.k,
        )

        judged = []
        binary_rels = []

        for item in retrieved:
            score, reason = judge_chunk(question, answer, item["text"])
            item_out = {
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "retrieval_score": item["retrieval_score"],
                "judge_score": score,
                "judge_reason": reason,
                "text": item["text"],
            }
            judged.append(item_out)
            binary_rels.append(1 if score >= 2 else 0)

        hit = 1 if any(binary_rels) else 0
        hit_count += hit

        rr = 0.0
        for i, rel in enumerate(binary_rels, start=1):
            if rel == 1:
                rr = 1.0 / i
                break
        mrr_total += rr

        precision_total += sum(binary_rels) / args.k

        gains = [j["judge_score"] for j in judged]
        actual_dcg = dcg(gains)
        ideal_dcg = dcg(sorted(gains, reverse=True))
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_total += ndcg

        result_row = {
            "id": qid,
            "question": question,
            "answer": answer,
            "hit@k_judge": hit,
            "rr": rr,
            "ndcg": ndcg,
            "judged_chunks": judged,
        }
        all_rows.append(result_row)

        print(f"{qid} hit@{args.k}={bool(hit)} rr={rr:.3f} ndcg={ndcg:.3f}")

        # save progress every question in case API stops halfway
        Path(args.out_path).write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    n = len(qa)
    summary = {
        f"Hit@{args.k}": hit_count / n,
        "MRR": mrr_total / n,
        f"Precision@{args.k}": precision_total / n,
        f"NDCG@{args.k}": ndcg_total / n,
        "num_questions": n,
    }

    final_output = {
        "summary": summary,
        "details": all_rows,
    }

    Path(args.out_path).write_text(
        json.dumps(final_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
