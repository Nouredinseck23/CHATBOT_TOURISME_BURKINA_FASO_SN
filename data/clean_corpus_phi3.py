import json, re, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- fichiers ----
RAW_DOCS = r"C:\Users\lenovo\Documents\HACKATON_SEMAINE_DU_NUMERIQUE\chatbot-tourisme-burkina\src\corpus_docs.json"
CLEAN_DOCS = r"C:\Users\lenovo\Documents\HACKATON_SEMAINE_DU_NUMERIQUE\chatbot-tourisme-burkina\src\corpus_docs_clean.json"

# ---- modèle local ----
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Chargement du modèle {MODEL_NAME} sur {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ---- nettoyage ----
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ").strip()
    return text

# ---- résumé avec Phi-3 ----
def summarize_phi3(text, max_len=512):
    prompt = f"Résume clairement ce texte en français, en gardant les points importants :\n\n{text}\n\nRésumé :"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Résumé :" in summary:
        summary = summary.split("Résumé :", 1)[-1].strip()
    return summary.strip()

# ---- pipeline ----
with open(RAW_DOCS, "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f" {len(docs)} documents chargés.")
cleaned_docs = []
seen = set()

for doc in tqdm(docs, desc="🧹 Nettoyage + résumé"):
    url = doc.get("url", "")
    text = clean_text(doc.get("text", ""))

    if not text or url in seen:
        continue
    seen.add(url)

    if len(text.split()) > 150:
        try:
            text = summarize_phi3(text)
        except Exception as e:
            print(f" Erreur de résumé pour {url}: {e}")
            continue

    cleaned_docs.append({"url": url, "text": text})

# ---- sauvegarde ----
with open(CLEAN_DOCS, "w", encoding="utf-8") as f:
    json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

print(f" Corpus nettoyé et résumé enregistré dans {CLEAN_DOCS}")
print(f" Nombre final de documents: {len(cleaned_docs)}")
