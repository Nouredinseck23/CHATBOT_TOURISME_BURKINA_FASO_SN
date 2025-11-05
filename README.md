1. Sujet choisi et justification

Pour ce mini-challenge, on a choisi de créer un chatbot sur le tourisme au Burkina Faso. L’idée, c’est de permettre à un utilisateur de poser des questions sur les sites touristiques, les musées, les festivals ou les villages artisanaux, et d’obtenir une réponse claire et sourcée.
On a choisi ce thème parce qu’il est à la fois riche en contenu et proche de notre culture. Le Burkina a un patrimoine très varié, et il existe déjà beaucoup d’informations disponibles en ligne (sur des sites comme l’ONTB ou BurkinaTourism) neanmoins au lieu de passer des minutes voir des heures a chercher des inbformations sur le tourisme burkinabe sur plusieurs de ces sites, l'utilisation d'un chatbot qui rassemble justement les informations de tout ces sites et est capable de donner des reponses en beaucoup moins de temps sans se coltiner par exmeple le spubs sur ces sites est une bien meilleure solution pour faciliter l'acces au information sur le tourisme burkinabe et contribuer a l'epanouissement de ce secteur.


2. Architecture technique

Notre projet suit une architecture RAG (Retrieval-Augmented Generation).
En gros, le système est divisé en trois grandes étapes :

Embeddings des textes → chaque passage de notre corpus est transformé en vecteur numérique avec le modèle multi-qa-mpnet-base-dot-v1.

Indexation avec FAISS → on stocke tous les vecteurs pour pouvoir retrouver rapidement les passages les plus proches d’une question.

Génération avec PHI-3 Mini → le modèle de langage génère une réponse à partir des passages trouvés, tout en essayant de citer les sources.

Pour rendre tout cela accessible, on a aussi ajouté une interface avec Gradio, afin que l’utilisateur puisse poser ses questions directement depuis une page web.
Et on a intégré FastAPI pour montrer qu’on pouvait aussi interagir avec le chatbot via une API (par exemple pour l’utiliser dans d’autres applications plus tard).

3. Technologies open source utilisées

Voici la liste des outils principaux qu’on a utilisés, tous 100 % open source :

Python 3.10+ → langage principal du projet

FAISS (Facebook AI Similarity Search) → pour la recherche vectorielle rapide
🔗 https://github.com/facebookresearch/faiss

SentenceTransformers → pour les embeddings avec le modèle multi-qa-mpnet-base-dot-v1
🔗 https://www.sbert.net

Microsoft PHI-3 Mini → modèle de génération léger et open source
🔗 https://huggingface.co/microsoft/phi-3-mini-4k-instruct

Gradio → pour l’interface utilisateur
🔗 https://gradio.app

4. Instructions d’installation
Prérequis :

Python installé (version ≥ 3.10)

Git et pip installés

Environ 4 Go de RAM minimum

Étapes :

Cloner le dépôt :

git clone https://github.com/nom-du-projet/chatbot-touristique.git
cd chatbot-touristique


Installer les dépendances :

pip install -r requirements.txt


Lancer l’application :

python interface.py


Ouvrir le lien local généré (exemple : http://127.0.0.1:7860/) pour accéder à l’interface Gradio.

5. Résultats et évaluation

On a testé le chatbot avec 20 questions sur le tourisme burkinabè (parcs, musées, festivals, etc.).
Dans la majorité des cas, le chatbot a donné des réponses base sur notre corpus et a su former des phares pour expliquer et repondre , surtout pour les questions simples.
 mais on a remarqué certaines limites : parfois le modèle hallucine (invente des réponses) ou cite mal les sources.
C’est normal, car le modèle PHI-3 Mini reste un petit modèle, donc il a ses faiblesses.
Malgré ça, notre système fonctionne bien pour une démo locale, et il prouve qu’on peut faire un assistant touristique 100 % open source et basé sur des données burkinabè.