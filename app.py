
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pytz

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from the .env file
load_dotenv()

# Function to configure the Google Generative AI API
def configure_api():
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("API key not found. Please update your .env file with a new GOOGLE_API_KEY.")
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring the API: {e}")
        return False

# This function loads the knowledge base and prepares the retrieval system.
@st.cache_resource
def setup_retrieval_system():
    knowledge_base = [
        {
            "tags": ["ganga aarti", "evening ceremony", "fire puja", "dashashwamedh ghat event", "when", "time", "where"],
            "context": "The Ganga Aarti is a spectacular fire worship ceremony at Dashashwamedh Ghat. It usually starts right after sunset, around 6:45 PM, and lasts for about 45 minutes. It's best to get there by 5:30 PM to find a good spot on the steps or on a boat."
        },
        {
            "tags": ["kashi vishwanath", "temple", "golden temple", "jyotirlinga", "darshan", "rules", "dress code"],
            "context": "The Kashi Vishwanath Temple is one of the most sacred Hindu temples, dedicated to Lord Shiva. Mobile phones, cameras, leather items (belts, wallets), and large bags are not allowed inside; free lockers are available. For men wishing to touch the Jyotirlinga (Sparsh Darshan), a traditional Dhoti-Kurta is required. For women, a Saree is required."
        },
        {
            "tags": ["sarnath", "buddhist site", "buddha sermon", "deer park", "history", "what to see", "distance", "how far"],
            "context": "Sarnath is a crucial Buddhist pilgrimage site about 10 km from Varanasi, where Lord Buddha gave his first sermon. Key sights include the massive Dhamek Stupa, the Ashokan Pillar, and an excellent archaeological museum. It takes about 30-45 minutes to reach by auto-rickshaw."
        },
        {
            "tags": ["boat ride", "boating", "ganges tour", "ghat view", "cost", "price", "sunrise"],
            "context": "A boat ride on the Ganges is a classic Varanasi experience, especially at sunrise. You can hire a private rowboat for around ₹300-₹500 per hour. It offers a stunning perspective of the ghats as the city wakes up."
        },
        {
            "tags": ["assi ghat", "morning", "yoga", "subah-e-banaras"],
            "context": "Assi Ghat is the southernmost main ghat, known for its lively and spacious atmosphere. It's famous for the 'Subah-e-Banaras' program, a daily morning event before sunrise that includes yoga, music, and a fire ceremony. It's a peaceful alternative to the evening Aarti."
        },
        {
            "tags": ["manikarnika ghat", "cremation", "funerals", "photography"],
            "context": "Manikarnika Ghat is the primary cremation ghat in Varanasi. It is a powerful place to witness Hindu funeral rites, where fires burn 24/7. It's a place of intense spirituality, and photography is strictly forbidden out of respect for the grieving families."
        },
        {
            "tags": ["transportation", "getting around", "auto rickshaw", "e-rickshaw", "ola", "uber"],
            "context": "For getting around Varanasi, e-rickshaws are great for short distances, especially in the crowded areas near the ghats. Auto-rickshaws are better for longer trips, like going to Sarnath or the airport. Always agree on the fare before starting your journey. Ride-sharing apps like Ola and Uber also work well, especially for airport transfers."
        },
        {
            "tags": ["shopping", "banarasi silk", "saree", "what to buy", "where to shop"],
            "context": "Varanasi is world-famous for its Banarasi silk sarees. The main areas for shopping are the Vishwanath Gali, Thatheri Bazar, and Godowlia Market. Besides sarees, you can buy wooden toys, stone carvings, and religious items. Bargaining is common in most shops."
        },
        {
            "tags": ["kachori sabzi", "best breakfast", "traditional food", "ram bhandar", "street food"],
            "context": "Kachori Sabzi is the quintessential Banarasi breakfast. It consists of a deep-fried, crisp pastry (kachori) served with a spicy potato curry (sabzi). The most legendary place to have it is Ram Bhandar in Thatheri Bazar."
        },
        {
            "tags": ["malaiyyo", "makhan malai", "winter sweet", "local dessert", "food"],
            "context": "Malaiyyo is a delicate and airy winter-exclusive dessert unique to Varanasi. It's made from milk foam flavored with saffron and cardamom. It's so light that it dissolves in your mouth and is sold by street vendors only in the early morning during the winter months (roughly November to February)."
        },
        {
            "tags": ["lassi", "blue lassi", "yogurt drink", "food"],
            "context": "Varanasi is famous for its lassi, a thick yogurt-based drink. While there are many shops, the Blue Lassi Shop, located in a narrow lane near Manikarnika Ghat, is a world-famous spot known for its wide variety of flavors served in clay cups (kulhads)."
        }
    ]
    
    retrieval_corpus = [" ".join(item['tags']) for item in knowledge_base]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(retrieval_corpus)
    
    return knowledge_base, vectorizer, tfidf_matrix

# This function generates the response from the AI.
def get_bot_response(user_query):
    ist = pytz.timezone('Asia/Kolkata')
    current_time_str = datetime.now(ist).strftime('%A, %B %d, %Y at %I:%M %p')

    user_vector = st.session_state.vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, st.session_state.tfidf_matrix)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]

    retrieved_context = ""
    if best_score > 0.1:
        retrieved_context = st.session_state.knowledge_base[best_match_index]['context']
        st.session_state.last_context = retrieved_context
    elif "last_context" in st.session_state:
        retrieved_context = st.session_state.last_context
    
    if not retrieved_context:
        return "I'm sorry, I don't seem to have information about that. Please ask me about the Ganga Aarti, Kashi Vishwanath Temple, Sarnath, or boat rides."

    # This standard model name WILL work with a new API key.
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    prompt = f"""
    You are Kashi Mitra, a friendly and expert tour guide for Varanasi.
    Your user is in Varanasi right now. The current date and time is {current_time_str}.
    Use this information to give helpful, context-aware answers. For example, if they ask about a winter-only food in a hot month, you should mention that it's not the right season.

    Using the following context, answer the user's question.

    Context: "{retrieved_context}"
    User's Question: "{user_query}"

    Your Answer:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"

# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(page_title="Kashi Mitra", page_icon="🙏", layout="wide")
st.title("🙏 Kashi Mitra - Your Varanasi Guide")

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image("varanasi.jpg", caption="Sunrise over the Ghats of Varanasi", width=600)

st.caption("I am an AI assistant powered by Google Gemini, here to help you explore the spiritual city of Varanasi.")

with st.sidebar:
    st.header("About Kashi Mitra")
    st.info("This chatbot uses AI to provide accurate, context-aware answers about Varanasi.")
    if st.button("Clear Conversation History"):
        st.session_state.messages = [{"role": "assistant", "content": "Namaste! How can I help you plan your visit to Varanasi today?"}]
        if "last_context" in st.session_state:
            del st.session_state.last_context
        st.rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Namaste! How can I help you plan your visit to Varanasi today?"}]
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base, st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_retrieval_system()

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. MAIN APP LOGIC ---

if configure_api():
    if prompt := st.chat_input("Ask me about temples, ghats, food..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_bot_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})