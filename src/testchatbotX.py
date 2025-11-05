from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- chemins ----
CORPUS_FILE = r"C:\Users\lenovo\Documents\HACKATON_SEMAINE_DU_NUMERIQUE\chatbot-tourisme-burkina\src\corpus_docs_clean.json"
EMBEDDINGS_FILE = r"C:\Users\lenovo\Documents\HACKATON_SEMAINE_DU_NUMERIQUE\chatbot-tourisme-burkina\src\corpus_embeddings_clean.npy"
FAISS_INDEX_FILE = r"C:\Users\lenovo\Documents\HACKATON_SEMAINE_DU_NUMERIQUE\chatbot-tourisme-burkina\src\vector_index_clean.faiss"

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

# ---- vérifier le GPU ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Utilisation du device : {device}")

# ---- charger le corpus nettoyé ----
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ---- découper les textes en passages cohérents ----
def split_paragraphs(docs, max_words=500):
    passages = []
    for doc in docs:
        text = doc["text"]
        url = doc.get("url", "")
        # découper par paragraphes
        for para in text.split("\n"):
            para = para.strip()
            if para:
                if len(para.split()) > max_words:
                    # si trop long, on le coupe en morceaux plus petits
                    words = para.split()
                    for i in range(0, len(words), max_words):
                        passages.append({"text": " ".join(words[i:i+max_words]), "url": url})
                else:
                    passages.append({"text": para, "url": url})
    return passages

passages = split_paragraphs(docs)
print(f"Nombre de passages après découpage : {len(passages)}")

# ---- créer embeddings ----
embed_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
texts = [p["text"] for p in passages]
embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# ---- sauvegarder embeddings ----
np.save(EMBEDDINGS_FILE, embeddings)

# ---- créer index FAISS ----
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # similarité cosinus
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_FILE)

# ---- charger modèle et tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# ---- fonction de recherche et génération ----
def answer_question(question, top_k=3, max_tokens=250):
    q_vec = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, top_k)

    # filtrer doublons
    seen_texts = set()
    context = ""
    for idx in indices[0]:
        text = passages[idx]["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            context += f"Source: {passages[idx]['url']}\n{text}\n\n"

    prompt = f"""
Tu es un guide touristique du Burkina Faso.
Réponds à la question suivante en utilisant uniquement les informations fournies.
Mentionne les sources entre parenthèses.
Fais une réponse concise et claire, regroupée par lieu.
Ne réécris pas les instructions, ne fais pas de résumé du résumé.

INFORMATIONS:
{context}

QUESTION: {question}

RÉPONSE:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(device)
    generation = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(generation[0], skip_special_tokens=True)
    return answer

# ---- test ----
if __name__ == "__main__":
    q = "qu'est-ce que je peux visiter à Ouagadougou ?"
    print(answer_question(q))
