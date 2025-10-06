# 🙏 Kashi Mitra - AI Chatbot for Varanasi

A sophisticated, AI-powered chatbot that acts as a virtual tour guide for the holy city of Varanasi, India.

---

###  Live Demo

 **You can chat with the live version of the bot here:**  https://kashi-mitra-chatbot.streamlit.app/
### 📋 About The Project

Kashi Mitra is a conversational AI built using a Retrieval-Augmented Generation (RAG) architecture. It provides accurate, context-aware information about Varanasi's famous ghats, temples, food, and culture. The goal is to offer a helpful and intelligent guide for tourists and pilgrims.


###  Features

* **Natural Conversation:** Ask questions in plain English.
* **Conversational Memory:** The bot remembers the last topic for natural follow-up questions.
* **Context-Aware:** It knows the current date and time to give smarter answers (e.g., about seasonal food).
* **Broad Knowledge Base:** Covers temples, ghats, food, shopping, and transportation.
* **Interactive Web UI:** Built with a clean and simple Streamlit interface.

---

###  Built With

* **Python**
* **Streamlit** - For the web interface
* **Google Gemini API** - For the generative AI model
* **Scikit-learn** - For the retrieval/search system

---

###  How to Run Locally

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/kashi-mitra-chatbot.git](https://github.com/your-username/kashi-mitra-chatbot.git)
    cd kashi-mitra-chatbot
    ```
2.  **Create a virtual environment and install dependencies:**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```
3.  **Create a `.env` file** and add your Google API Key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
4.  **Run the app:**
    ```sh
    streamlit run app.py
    ```
