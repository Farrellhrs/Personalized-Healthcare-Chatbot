import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
import json

# Import custom modules
from auth_handler import AuthHandler
from intent_classifier import IntentClassifier
from database_handler import DatabaseHandler
from llm_handler import LLMHandler
from recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration with enhanced mobile support
st.set_page_config(
    page_title="CarePal - Asisten Kesehatan AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "CarePal - Asisten AI untuk mendampingi perjalanan kesehatan Anda. Mendukung kehamilan dan kesehatan umum."
    }
)

# Enhanced CSS for modern UI design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles - Light Mode */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.2;
    }
    
    /* Welcome header */
    .welcome-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #34495e;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Recommendation buttons - Light Mode */
    .recommendation-button {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 4px;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        font-weight: 500;
        text-align: left;
    }
    
    .recommendation-button:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Chat container - Light Mode */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Chat message base styling */
    .chat-message {
        display: flex;
        flex-direction: column;
        margin: 0.8rem 0;
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User message styling - Light Mode */
    .user-message {
        align-items: flex-end;
    }
    
    .user-bubble {
        background: #667eea;
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        font-size: 0.95rem;
        line-height: 1.4;
        word-wrap: break-word;
    }
    
    .user-label {
        color: #667eea;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 3px;
        margin-right: 10px;
        line-height: 1;
    }
    
    /* Assistant message styling - Light Mode */
    .bot-message {
        align-items: flex-start;
    }
    
    .bot-bubble {
        background: #f8f9fa;
        color: #2c3e50;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 0.95rem;
        line-height: 1.4;
        word-wrap: break-word;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .bot-label {
        color: #667eea;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 3px;
        margin-left: 10px;
        line-height: 1;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.7rem;
        color: #95a5a6;
        margin-top: 5px;
        font-weight: 300;
    }
    
    /* Send button styling */
    .send-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .send-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            line-height: 1.3;
        }
        
        .welcome-header {
            font-size: 1.5rem;
        }
        
        .user-bubble, .bot-bubble {
            max-width: 95%;
            font-size: 0.9rem;
        }
        
        .chat-container {
            max-height: 50vh;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.6rem;
            margin-bottom: 1.5rem;
        }
        
        .welcome-header {
            font-size: 1.3rem;
        }
        
        .recommendation-button {
            margin: 5px 0;
            font-size: 0.85rem;
        }
    }
    
    /* Custom scrollbar - Light Mode */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #5a6fd8;
    }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        color: #667eea;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "contextual_recommendations" not in st.session_state:
        st.session_state.contextual_recommendations = []
    if "message_input" not in st.session_state:
        st.session_state.message_input = ""
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

def clean_markdown_text(text):
    """Remove markdown formatting from text"""
    import re
    
    # Remove bold markdown (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove italic markdown (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove other common markdown patterns
    text = re.sub(r'`(.*?)`', r'\1', text)  # Code blocks
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)  # Headers
    
    return text.strip()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    try:
        base_path = r"C:\Users\farre\Documents\Kuliah\Magang era\Project 1"
        
        auth_handler = AuthHandler(base_path)
        intent_classifier = IntentClassifier(base_path)
        database_handler = DatabaseHandler(base_path)
        llm_handler = LLMHandler()  # Will need API key
        recommendation_engine = RecommendationEngine(base_path)
        
        return auth_handler, intent_classifier, database_handler, llm_handler, recommendation_engine
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None, None, None, None, None

def login_page():
    """Display modern login page"""
    st.markdown('<h1 class="main-header">🏥 CarePal - Asisten Kesehatan AI</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            nik = st.text_input("NIK", placeholder="Masukkan NIK Anda", help="Nomor Induk Kependudukan")
            password = st.text_input("Password", type="password", placeholder="Masukkan Password", help="Kata sandi akun Anda")
            
            # Add some spacing
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            
            submit_button = st.form_submit_button("Masuk 🔐", use_container_width=True)
            
            if submit_button:
                if not nik or not password:
                    st.error("NIK dan Password harus diisi!")
                else:
                    # Authenticate user
                    auth_handler, _, _, _, _ = initialize_components()
                    if auth_handler:
                        user_data = auth_handler.authenticate(nik, password)
                        if user_data:
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            st.success("Login berhasil! Mengalihkan ke CarePal...")
                            st.rerun()
                        else:
                            st.error("NIK atau Password salah!")
                    else:
                        st.error("Sistem tidak dapat diinisialisasi. Silakan coba lagi.")

def chatbot_page():
    """Main chatbot interface"""
    auth_handler, intent_classifier, database_handler, llm_handler, recommendation_engine = initialize_components()
    
    if not all([auth_handler, intent_classifier, database_handler, llm_handler, recommendation_engine]):
        st.error("Sistem tidak dapat diinisialisasi dengan lengkap. Silakan refresh halaman.")
        return
    
    # Clean sidebar without extra containers
    with st.sidebar:
        st.markdown("""
        <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center; font-weight: 600;">
            👋 Selamat Datang!
        </h3>
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid rgba(102, 126, 234, 0.2);">
            <p style="color: #2c3e50; margin: 0.5rem 0; font-weight: 500;">
                <span style="opacity: 0.7; font-weight: 400;">Nama:</span><br>
                <strong>{name}</strong>
            </p>
            <p style="color: #2c3e50; margin: 0.5rem 0; font-weight: 500;">
                <span style="opacity: 0.7; font-weight: 400;">NIK:</span><br>
                <strong>{nik}</strong>
            </p>
        </div>
        """.format(
            name=st.session_state.user_data['name'],
            nik=st.session_state.user_data['NIK']
        ), unsafe_allow_html=True)
        
        if st.button("Keluar 🚪", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.chat_history = []
            st.session_state.recommendations = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <h4 style="color: #2c3e50; margin-bottom: 1rem; font-weight: 600;">
            💬 Riwayat Chat
        </h4>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface with modern welcome
    st.markdown(f'''
    <div class="welcome-header">
        Halo, {st.session_state.user_data["name"]}! 👋
    </div>
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="color: #7f8c8d; font-size: 1rem; margin: 0;">
            Saya CarePal, siap membantu Anda dengan pertanyaan seputar kesehatan
        </p>
        <p style="color: #95a5a6; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Mendukung konsultasi kehamilan dan kesehatan umum
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Generate recommendations if not already done
    if not st.session_state.recommendations:
        with st.spinner("Memuat rekomendasi pertanyaan..."):
            try:
                if recommendation_engine and database_handler and llm_handler:
                    recommendations = recommendation_engine.generate_recommendations(
                        st.session_state.user_data['customer_id'],
                        database_handler,
                        llm_handler
                    )
                    st.session_state.recommendations = recommendations
                else:
                    # Fallback recommendations if components are not available
                    st.session_state.recommendations = [
                        "Apa hasil lab terakhir saya?",
                        "Diagnosis terakhir saya apa?",
                        "Kapan jadwal kontrol berikutnya?",
                        "Bagaimana kondisi kesehatan saya saat ini?"
                    ]
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.session_state.recommendations = [
                    "Apa hasil lab terakhir saya?",
                    "Diagnosis terakhir saya apa?",
                    "Kapan jadwal kontrol berikutnya?",
                    "Bagaimana kondisi kesehatan saya saat ini?"
                ]
    
    # Clean recommendations section
    st.markdown("""
    <h3 style="
        color: #2c3e50;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
        font-size: 1.2rem;
    ">💡 Pertanyaan yang Mungkin Anda Butuhkan</h3>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, recommendation in enumerate(st.session_state.recommendations[:4]):
        col_index = i % 2
        with cols[col_index]:
            if st.button(
                recommendation, 
                key=f"rec_{i}", 
                use_container_width=True,
                help="Klik untuk menggunakan pertanyaan ini"
            ):
                st.session_state.message_input = recommendation
                process_user_message(recommendation, intent_classifier, database_handler, llm_handler)
    
    st.markdown("---")
    
    # Clean chat history display with contextual recommendations
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.chat_history):
            # Generate timestamp
            timestamp = datetime.now().strftime("%H:%M")
            
            # Clean the message content from any markdown
            clean_content = clean_markdown_text(message["content"])
            
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <div class="user-label">Anda</div>
                    <div class="user-bubble">{clean_content}</div>
                    <div class="timestamp" style="text-align: right; margin-right: 10px;">{timestamp}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message bot-message">
                    <div class="bot-label">CarePal</div>
                    <div class="bot-bubble">{clean_content}</div>
                    <div class="timestamp" style="margin-left: 10px;">{timestamp}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show contextual recommendations after bot response
                if i == len(st.session_state.chat_history) - 1 and st.session_state.contextual_recommendations:
                    st.markdown('''
                    <div style="margin: 1rem 0; padding: 0 10px;">
                        <div style="color: #667eea; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">
                            💡 Pertanyaan lanjutan yang mungkin Anda butuhkan:
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Display contextual recommendations as compact buttons
                    cols = st.columns(2)
                    for idx, rec in enumerate(st.session_state.contextual_recommendations[:4]):
                        col_index = idx % 2
                        with cols[col_index]:
                            if st.button(
                                rec, 
                                key=f"contextual_rec_{i}_{idx}", 
                                use_container_width=True,
                                help="Klik untuk melanjutkan dengan pertanyaan ini",
                                type="secondary"
                            ):
                                st.session_state.message_input = rec
                                process_user_message(rec, intent_classifier, database_handler, llm_handler)
                                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple input area
    st.markdown("### ✍️ Ketik Pertanyaan Anda")
    
    with st.form("message_form", clear_on_submit=True):
        user_input = st.text_area(
            "Pertanyaan Anda:", 
            value=st.session_state.message_input, 
            height=100, 
            placeholder="Ketik pertanyaan Anda di sini... Contoh: 'Bagaimana hasil lab terakhir saya?' atau 'Kapan jadwal kontrol berikutnya?'",
            help="Gunakan bahasa Indonesia yang jelas dan spesifik untuk hasil terbaik"
        )
        
        col1, col2 = st.columns([4, 1])
        
        with col2:
            send_button = st.form_submit_button("Kirim 📤", use_container_width=True)
        
        if send_button and user_input.strip():
            st.session_state.message_input = ""
            process_user_message(user_input.strip(), intent_classifier, database_handler, llm_handler)
            st.rerun()

def process_user_message(user_input, intent_classifier, database_handler, llm_handler):
    """Process user message through the pipeline with modern UI feedback"""
    try:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show modern loading indicator
        with st.spinner("🤖 Memproses pertanyaan Anda..."):
            # Step 1: Check similarity with intent_merged.csv
            similarity_result = intent_classifier.check_similarity(user_input)
            
            if not similarity_result["is_valid"]:
                response = "Maaf, itu di luar fitur saya. Saya CarePal dapat membantu dengan pertanyaan seputar kesehatan kehamilan dan umum. Silakan coba pertanyaan lain yang berkaitan dengan kesehatan Anda."
            else:
                # Step 2: Domain classification (KEHAMILAN vs UMUM)
                domain = intent_classifier.classify_domain(user_input)
                
                # Step 3: Get specific intent
                if domain == "KEHAMILAN":
                    intent_result = intent_classifier.classify_pregnancy_intent(user_input)
                else:
                    intent_result = intent_classifier.classify_general_intent(user_input)
                
                # Step 4: Fetch relevant database data for top predictions
                customer_id = st.session_state.user_data['customer_id']
                
                # Get primary context (backward compatibility)
                primary_context = database_handler.get_context_for_intent(
                    intent_result["intent"], 
                    customer_id
                )
                
                # Get contexts for top 2 predictions if available
                contexts = {'primary': primary_context}
                if 'predictions' in intent_result and len(intent_result['predictions']) > 1:
                    for i, prediction in enumerate(intent_result['predictions'][:2]):
                        context_key = f"prediction_{i+1}"
                        contexts[context_key] = database_handler.get_context_for_intent(
                            prediction['intent'],
                            customer_id
                        )
                
                # Step 5: Generate response using LLM
                response = llm_handler.generate_response(
                    user_input=user_input,
                    intent=intent_result["intent"],
                    confidence=intent_result["confidence"],
                    db_context=primary_context,  # Keep primary for backward compatibility
                    user_data=st.session_state.user_data,
                    top_predictions=intent_result.get("predictions", []),
                    contexts=contexts  # New: All contexts for top predictions
                )
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Generate contextual recommendations for next questions
        if similarity_result["is_valid"]:
            # Update session state with conversation context
            st.session_state.last_intent = intent_result["intent"]
            st.session_state.last_user_input = user_input
            
            # Generate contextual recommendations
            try:
                recommendation_engine = RecommendationEngine(".")
                contextual_recs = recommendation_engine.get_contextual_recommendations(
                    intent=intent_result["intent"],
                    customer_id=customer_id,
                    database_handler=database_handler,
                    user_input=user_input,
                    response_content=response
                )
                
                # Update session state with contextual recommendations
                st.session_state.contextual_recommendations = contextual_recs
            except Exception as e:
                logger.warning(f"Could not generate contextual recommendations: {str(e)}")
                st.session_state.contextual_recommendations = []
        else:
            # Clear recommendations for invalid queries
            st.session_state.contextual_recommendations = []
        
        # Show success indicator briefly
        st.success("✅ Respons berhasil dihasilkan!")
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_response = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi dalam beberapa saat atau hubungi administrator jika masalah berlanjut."
        st.session_state.chat_history.append({"role": "assistant", "content": error_response})
        st.error("❌ Terjadi kesalahan dalam memproses pesan")

def main():
    """Main application function"""
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chatbot_page()

if __name__ == "__main__":
    main()
