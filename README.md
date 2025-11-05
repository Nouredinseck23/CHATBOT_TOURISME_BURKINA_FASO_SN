Rapport de Projet : Chatbot Thématique - Tourisme au Burkina FasoCe rapport présente la conception, l'architecture et les résultats de notre mini-challenge visant à créer un assistant conversationnel pour la promotion du tourisme au Burkina Faso.1. Sujet et Justification du Choix🎯 Sujet ChoisiNous avons développé un chatbot spécialisé dans le tourisme au Burkina Faso. Cet outil permet aux utilisateurs d'interroger directement le système sur des sujets variés tels que :Sites touristiques majeurs.Musées et lieux culturels.Festivals nationaux.Villages artisanaux.💡 Justification et Valeur AjoutéeLe Burkina Faso possède un patrimoine culturel et naturel exceptionnellement riche. Bien que des informations existent sur des plateformes comme l'ONTB ou BurkinaTourism, la recherche d'informations précises est souvent longue et dispersée.Notre chatbot propose une solution d'accès à l'information beaucoup plus rapide et centralisée. En agrégeant et en rendant accessible ce corpus via une interface conversationnelle, nous facilitons grandement la découverte du tourisme burkinabè, contribuant ainsi à l'épanouissement de ce secteur.2. Architecture Technique : RAG (Retrieval-Augmented Generation)Notre projet repose sur une architecture moderne de Génération Augmentée par Récupération (RAG), garantissant des réponses pertinentes et sourcées.Le processus est structuré en trois phases clés :🔍 Embeddings des Textes : Chaque segment de notre corpus documentaire est converti en un vecteur numérique (embedding) à l'aide du modèle performant multi-qa-mpnet-base-dot-v1.🗂️ Indexation Vectorielle (FAISS) : Les vecteurs sont stockés et indexés dans une base de données FAISS (Facebook AI Similarity Search). Cela permet une recherche ultra-rapide pour identifier les passages du corpus les plus pertinents par rapport à la question de l'utilisateur.🗣️ Génération de la Réponse (PHI-3 Mini) : Le modèle de langage (LLM) utilise les passages de texte récupérés pour générer une réponse cohérente, factuelle et contextuelle, s'efforçant toujours de citer les sources d'origine.Pour l'interaction utilisateur, nous avons mis en place :Une interface web conviviale via Gradio.Une API d'interaction via FastAPI, pour une intégration future dans d'autres applications.3. Technologies Open Source UtiliséesCe projet est construit entièrement avec des outils 100 % Open Source :ComposantTechnologieRôle PrincipalLienLangage PrincipalPython $\geq 3.10$Développement du projet.N/ARecherche VectorielleFAISS (Facebook AI Similarity Search)Indexation et recherche rapide des embeddings.🔗 GitHubGénération d'EmbeddingsSentenceTransformersModèle multi-qa-mpnet-base-dot-v1 pour les représentations vectorielles.🔗 Site WebModèle de GénérationMicrosoft PHI-3 MiniLLM léger pour la création de réponses.🔗 Hugging FaceInterface UtilisateurGradioCréation d'une démo web interactive.🔗 Site Web4. Instructions d'Installation et de Démarrage🛠️ PrérequisPython (version $\geq 3.10$)Git et pip installésMinimum 4 Go de RAM📄 Liste des Dépendances (requirements.txt)Voici la liste des paquets Python nécessaires à l'exécution du projet, installables via pip install -r requirements.txt :# === Core Python ===
numpy
pandas
tqdm

# === NLP / Embeddings ===
sentence-transformers    # Pour les embeddings (ex: multi-qa-mpnet-base-dot-v1)
transformers              # Pour le modèle génératif (Phi-3-mini)
torch                     # Backend pour les modèles HF

# === Vector Store / Similarity Search ===
faiss-cpu                # Recherche vectorielle rapide
chromadb                 # Gestion du stockage des embeddings et documents

# === Frontend (Interface) ===
gradio                   # Interface web utilisateur du chatbot

# === Evaluation / Metrics ===
scikit-learn             # Calcul précision / métriques
nltk                     # Nettoyage texte / tokenisation

# === Data Processing / Scraping ===
beautifulsoup4           # Parsing HTML
requests                 # Téléchargement des pages web
lxml                     # Parser XML/HTML rapide
🚀 Étapes de LancementCloner le dépôt :git clone [https://github.com/nom-du-projet/chatbot-touristique.git](https://github.com/nom-du-projet/chatbot-touristique.git)
cd chatbot-touristique

Installer les dépendances :pip install -r requirements.txt

Lancer l'application :python interface.py

Accéder à l'interface :Ouvrez le lien local généré (ex. : http://127.0.0.1:7860/) dans votre navigateur pour interagir avec le chatbot via Gradio.5. Résultats et Évaluation✅ PerformancesLe chatbot a été testé avec une batterie de 20 questions portant sur le tourisme burkinabè (parcs, musées, festivals, etc.).Réponses factuelles : Pour la majorité des requêtes simples, le système a démontré sa capacité à formuler des réponses claires et basées sur le corpus, prouvant la fonctionnalité de l'architecture RAG.Preuve de concept : Le projet valide la faisabilité de créer un assistant touristique local, performant et 100 % open source, en s'appuyant sur des données burkinabè.⚠️ LimitesConformément aux attentes pour un modèle de sa taille, le PHI-3 Mini a montré certaines faiblesses inhérentes aux petits LLM :Hallucinations : Une tendance occasionnelle à inventer des informations.Imprécisions de Sourçage : Des citations de sources parfois incorrectes ou incomplètes.Ces limites n'empêchent pas le système de fonctionner efficacement comme une preuve de concept et une démo locale solide.
