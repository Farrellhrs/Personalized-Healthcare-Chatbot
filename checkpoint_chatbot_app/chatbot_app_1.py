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

# Page configuration
st.set_page_config(
    page_title="Chatbot Kesehatan Ibu Hamil",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for modern, professional UI design
st.markdown("""
<style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1000px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2D3748;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Welcome message */
    .welcome-message {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4A5568;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Recommendation buttons */
    .recommendation-section {
        background: #F7FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
    }
    
    .recommendation-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2D3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    
    /* Chat message bubbles */
    .chat-message {
        display: flex;
        margin: 1rem 0;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .user-message {
        flex-direction: row-reverse;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 0.875rem 1.125rem;
        border-radius: 18px;
        font-size: 0.9rem;
        line-height: 1.4;
        word-wrap: break-word;
        position: relative;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-bubble {
        background: #FFFFFF;
        color: #2D3748;
        border: 1px solid #E2E8F0;
        border-bottom-left-radius: 4px;
    }
    
    /* Message avatar/icon */
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .assistant-avatar {
        background: #F7FAFC;
        color: #667eea;
        border: 2px solid #E2E8F0;
    }
    
    /* Message header */
    .message-header {
        font-size: 0.75rem;
        color: #718096;
        margin-bottom: 0.25rem;
        font-weight: 500;
    }
    
    /* Timestamp */
    .message-timestamp {
        font-size: 0.7rem;
        color: #A0AEC0;
        margin-top: 0.375rem;
        text-align: right;
    }
    
    .user-message .message-timestamp {
        text-align: left;
    }
    
    /* Input area styling */
    .input-container {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: sticky;
        bottom: 0;
        z-index: 100;
    }
    
    .input-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2D3748;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sidebar-info h3 {
        margin-top: 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #E2E8F0, transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {
        display: none;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #CBD5E0;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #A0AEC0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .message-bubble {
            max-width: 85%;
        }
        
        .main-header {
            font-size: 1.8rem;
        }
        
        .sidebar-info {
            padding: 1rem;
        }
    }
    
    /* Loading animation */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #CBD5E0;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
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
    if "message_input" not in st.session_state:
        st.session_state.message_input = ""

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
    st.markdown('''
    <div class="main-header">Chatbot Kesehatan Ibu Hamil</div>
    <div style="text-align: center; margin-bottom: 3rem; color: #718096;">
        <p>Sistem informasi kesehatan untuk ibu hamil dengan teknologi AI</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('''
        <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #E2E8F0;">
            <h3 style="text-align: center; margin-bottom: 1.5rem; color: #2D3748;">Masuk ke Akun Anda</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        with st.form("login_form"):
            nik = st.text_input(
                "NIK", 
                placeholder="Masukkan NIK Anda (16 digit)",
                help="Nomor Induk Kependudukan sesuai KTP"
            )
            password = st.text_input(
                "Password", 
                type="password", 
                placeholder="Masukkan Password",
                help="Password untuk akses sistem"
            )
            
            col_login1, col_login2, col_login3 = st.columns([1, 1, 1])
            with col_login2:
                submit_button = st.form_submit_button(
                    "Masuk", 
                    use_container_width=True,
                    type="primary"
                )
            
            if submit_button:
                if not nik or not password:
                    st.error("🚨 NIK dan Password harus diisi!")
                elif len(nik) != 16:
                    st.error("🚨 NIK harus terdiri dari 16 digit!")
                else:
                    # Authenticate user with modern loading
                    with st.spinner("🔐 Memverifikasi kredensial..."):
                        auth_handler, _, _, _, _ = initialize_components()
                        if auth_handler:
                            user_data = auth_handler.authenticate(nik, password)
                            if user_data:
                                st.session_state.authenticated = True
                                st.session_state.user_data = user_data
                                st.success("✅ Login berhasil! Mengalihkan ke chatbot...")
                                st.rerun()
                            else:
                                st.error("❌ NIK atau Password salah! Silakan coba lagi.")
                        else:
                            st.error("⚠️ Sistem tidak dapat diinisialisasi. Silakan refresh halaman.")
        
        # Footer information
        st.markdown('''
        <div style="text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #A0AEC0;">
            <p>Sistem ini menggunakan teknologi AI untuk membantu informasi kesehatan</p>
            <p>🔒 Data Anda aman dan terlindungi</p>
        </div>
        ''', unsafe_allow_html=True)

def format_message_bubble(message_content, is_user=True, timestamp=None):
    """
    Format a chat message bubble with modern styling (no raw markdown)
    
    Args:
        message_content (str): The message content
        is_user (bool): True if user message, False if assistant message
        timestamp (str): Optional timestamp string
        
    Returns:
        str: HTML formatted message bubble
    """
    # Get current timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    # Clean any raw markdown formatting from message content
    clean_content = message_content.replace('**', '').replace('*', '')
    
    if is_user:
        return f"""
        <div class="chat-message user-message">
            <div class="message-bubble user-bubble">
                <div class="message-header">Anda</div>
                {clean_content}
                <div class="message-timestamp">{timestamp}</div>
            </div>
            <div class="message-avatar user-avatar">👤</div>
        </div>
        """
    else:
        return f"""
        <div class="chat-message">
            <div class="message-avatar assistant-avatar">🤖</div>
            <div class="message-bubble assistant-bubble">
                <div class="message-header">Assistant</div>
                {clean_content}
                <div class="message-timestamp">{timestamp}</div>
            </div>
        </div>
        """

def chatbot_page():
    """Main chatbot interface"""
    auth_handler, intent_classifier, database_handler, llm_handler, recommendation_engine = initialize_components()
    
    if not all([auth_handler, intent_classifier, database_handler, llm_handler, recommendation_engine]):
        st.error("Sistem tidak dapat diinisialisasi dengan lengkap. Silakan refresh halaman.")
        return
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown('''
        <div class="sidebar-info">
            <h3>Selamat Datang!</h3>
            <p><strong>Nama:</strong> {}</p>
            <p><strong>NIK:</strong> {}</p>
        </div>
        '''.format(st.session_state.user_data['name'], st.session_state.user_data['NIK']), unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True, type="primary"):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.chat_history = []
            st.session_state.recommendations = []
            st.rerun()
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("**Riwayat Chat**")
        if st.button("Bersihkan Riwayat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface with modern header
    st.markdown(f'''
    <div class="main-header">Chatbot Kesehatan Ibu Hamil</div>
    <div class="welcome-message">Halo, {st.session_state.user_data["name"]}! 👋 Saya siap membantu Anda.</div>
    ''', unsafe_allow_html=True)
    
    # Generate recommendations if not already done
    if not st.session_state.recommendations:
        with st.spinner("Memuat rekomendasi pertanyaan..."):
            try:
                recommendations = recommendation_engine.generate_recommendations(
                    st.session_state.user_data['customer_id'],
                    database_handler,
                    llm_handler
                )
                st.session_state.recommendations = recommendations
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.session_state.recommendations = [
                    "Apa hasil lab terakhir saya?",
                    "Diagnosis terakhir saya apa?",
                    "Kapan jadwal kontrol berikutnya?",
                    "Obat apa yang diresepkan untuk saya?"
                ]
    
    # Display recommendations with modern styling
    st.markdown('''
    <div class="recommendation-section">
        <div class="recommendation-title">💡 Pertanyaan yang Mungkin Anda Butuhkan</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Create recommendation buttons in a clean layout
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
                # Add timestamp to user message
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": recommendation,
                    "timestamp": timestamp
                })
                process_user_message(recommendation, intent_classifier, database_handler, llm_handler)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # Chat history with modern message bubbles
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            is_user = message["role"] == "user"
            timestamp = message.get("timestamp", datetime.now().strftime("%H:%M"))
            
            # Format message bubble without raw markdown
            formatted_message = format_message_bubble(
                message["content"], 
                is_user=is_user, 
                timestamp=timestamp
            )
            st.markdown(formatted_message, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="text-align: center; padding: 2rem; color: #718096;">
            <p>Belum ada percakapan. Mulai dengan mengetik pertanyaan atau pilih salah satu rekomendasi di atas! 💬</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Modern input area
    st.markdown('''
    <div class="input-container">
        <div class="input-title">✍️ Ketik Pertanyaan Anda</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Create form with modern styling
    with st.form("message_form", clear_on_submit=True):
        user_input = st.text_area(
            label="Pertanyaan Anda:",
            value=st.session_state.message_input,
            height=100,
            placeholder="Ketik pertanyaan Anda di sini... (contoh: Kapan jadwal kontrol ANC saya berikutnya?)",
            label_visibility="collapsed",
            key="user_input_area"
        )
        
        # Modern send button layout
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col3:
            send_button = st.form_submit_button(
                "Kirim", 
                use_container_width=True,
                type="primary"
            )
        
        if send_button and user_input.strip():
            st.session_state.message_input = ""
            # Add timestamp to user message
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_input.strip(),
                "timestamp": timestamp
            })
            process_user_message(user_input.strip(), intent_classifier, database_handler, llm_handler)
            st.rerun()

def process_user_message(user_input, intent_classifier, database_handler, llm_handler):
    """Process user message through the pipeline with modern UI feedback"""
    try:
        # Show typing indicator
        with st.spinner("🤔 Memproses pertanyaan Anda..."):
            # Step 1: Check similarity with intent_merged.csv
            similarity_result = intent_classifier.check_similarity(user_input)
            
            if not similarity_result["is_valid"]:
                response = "Maaf, itu di luar fitur saya. Saya hanya dapat membantu dengan pertanyaan seputar kehamilan dan data kesehatan Anda."
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
        
        # Add bot response to history with timestamp
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_response = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi."
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_response,
            "timestamp": timestamp
        })

def main():
    """Main application function"""
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chatbot_page()

if __name__ == "__main__":
    main()
