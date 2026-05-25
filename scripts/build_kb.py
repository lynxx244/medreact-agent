import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_kb(
    raw_path="data/test_datasets.jsonl",
    testset_path="data/testset_labeled_llm.jsonl",
    kb_dir="kb",
    max_docs=6000,
):
    test_ids = {json.loads(line)["id"] for line in open(testset_path, encoding="utf-8")}

    questions = []
    answers = []
    for idx, line in enumerate(open(raw_path, encoding="utf-8")):
        if len(questions) >= max_docs:
            break
        if idx in test_ids:
            continue
        raw = json.loads(line)
        q = raw.get("questions", "").strip()
        a = raw.get("answers", "").strip()
        if q and a:
            questions.append(q)
            answers.append(a)

    model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    vectors = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    Path(kb_dir).mkdir(exist_ok=True)
    faiss.write_index(index, f"{kb_dir}/kb.index")
    with open(f"{kb_dir}/answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)

    print(f"知识库构建完成，共 {len(answers)} 条")

if __name__ == "__main__":
    build_kb()
