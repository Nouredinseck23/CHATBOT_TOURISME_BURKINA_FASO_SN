# interface.py
import gradio as gr
from testchatbotX import answer_question  # on importe notre fonction existante

# Fonction d’interface
def chatbot_gradio(question):
    
    return answer_question(question)

# Création de l’interface
iface = gr.Interface(
    fn=chatbot_gradio,          
    inputs=gr.Textbox(label="Pose ta question"), 
    outputs=gr.Textbox(label="Réponse du chatbot"), 
    title="Chatbot Tourisme Burkina Faso 🇧🇫",
    description="Posez votre question sur le tourisme au Burkina Faso et obtenez une réponse concise."
)

# Lancer l'interface
if __name__ == "__main__":
    iface.launch()
