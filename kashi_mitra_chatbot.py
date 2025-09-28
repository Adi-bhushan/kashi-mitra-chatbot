# Kashi Mitra - AI Chatbot for Varanasi

# --- STAGE 4: Generative AI with RAG (Retrieval-Augmented Generation) ---
# We connect our smart search with a powerful Large Language Model (LLM).

import os
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generative_ai_chatbot():
    """
    This chatbot uses RAG to provide accurate, conversational answers.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Configure the Google Generative AI with the API key
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please create a .env file with GOOGLE_API_KEY.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring the API: {e}")
        return

    # --- KNOWLEDGE BASE (Our Factual Data) ---
    knowledge_base = [
        {
            "tags": ["ganga aarti", "evening ceremony", "fire puja", "dashashwamedh ghat event", "when", "time", "where"],
            "context": "The Ganga Aarti is a spectacular fire worship ceremony conducted daily on the banks of the Ganges River. The most famous one is at Dashashwamedh Ghat. It usually starts right after sunset, around 6:30 PM to 7:00 PM, and lasts for about 45 minutes. Thousands of people gather on the ghats and on boats to witness it."
        },
        {
            "tags": ["kachori sabzi", "best breakfast", "traditional food", "ram bhandar", "street food", "where to eat"],
            "context": "Kachori Sabzi is a quintessential Banarasi breakfast. It consists of a deep-fried, crisp pastry (kachori) served with a spicy potato curry (sabzi). The most legendary place to have it is Ram Bhandar in Thatheri Bazar, which has been serving it for generations. Expect a crowd."
        },
        {
            "tags": ["sarnath", "buddhist site", "buddha sermon", "deer park", "history", "what to see"],
            "context": "Sarnath is a crucial Buddhist pilgrimage site located about 10 km northeast of Varanasi. It's famous because it is where Lord Buddha delivered his first sermon after attaining enlightenment. Key sights include the Dhamek Stupa, the Chaukhandi Stupa, and the Ashokan Pillar. The Sarnath Museum is also highly recommended."
        },
        {
            "tags": ["kashi vishwanath", "temple", "golden temple", "jyotirlinga", "darshan", "rules"],
            "context": "The Kashi Vishwanath Temple is one of the most sacred Hindu temples, dedicated to Lord Shiva. It is one of the twelve Jyotirlingas. The temple has a striking golden spire. For darshan (viewing), mobile phones, cameras, and large bags are not allowed inside. There are free lockers available nearby. Be prepared for long queues, especially on Mondays."
        }
    ]

    # --- RETRIEVAL SETUP (Same as Stage 3) ---
    all_contexts = [item['context'] for item in knowledge_base]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_contexts)
    
    # --- GENERATION SETUP (The LLM) ---
    model = genai.GenerativeModel('gemini-1.5-flash')

    print("Namaste! I am Kashi Mitra (v3.0 - Powered by Gemini). I can now have a more detailed conversation.")

    while True:
        user_input = input("You: ").lower()

        if user_input in ["bye", "exit", "quit"]:
            print("Kashi Mitra: Farewell! May your journey in Varanasi be enlightening.")
            break

        # 1. RETRIEVAL: Find the most relevant context from our knowledge base
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        best_match_index = similarities.argmax()
        best_score = similarities[0, best_match_index]

        if best_score > 0.15: # Threshold to ensure we have some relevant info
            retrieved_context = knowledge_base[best_match_index]['context']
            
            # 2. AUGMENTED GENERATION: Create a prompt for the LLM
            prompt = f"""
            You are Kashi Mitra, a friendly and expert tour guide for Varanasi.
            Using the following context, answer the user's question in a conversational and helpful way.
            Do not provide information that is not in the context.

            Context: "{retrieved_context}"
            
            User's Question: "{user_input}"

            Your Answer:
            """
            
            # Generate the response
            try:
                response = model.generate_content(prompt)
                print(f"Kashi Mitra: {response.text}")
            except Exception as e:
                print(f"Kashi Mitra: Sorry, I encountered an error while generating a response: {e}")

        else:
            print("Kashi Mitra: I'm sorry, I don't seem to have information about that. Please ask me about the Ganga Aarti, Kashi Vishwanath Temple, Sarnath, or local food like Kachori Sabzi.")

if __name__ == "__main__":
    generative_ai_chatbot()