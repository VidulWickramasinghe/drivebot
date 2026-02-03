import streamlit as st
import requests
from pathlib import Path
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
INGEST_ENDPOINT = f"{API_BASE_URL}/ingest"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Page config
st.set_page_config(
    page_title="AutoMentor - Automotive AI Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None

def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("AutoMentor_initialized", False)
        return False
    except requests.exceptions.RequestException:
        return False

def send_query(question: str):
    """Send a query to the AutoMentor API."""
    try:
        response = requests.post(
            QUERY_ENDPOINT,
            json={"question": question},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}\n\nMake sure the API server is running:\nuvicorn api.server:app --host 0.0.0.0 --port 8000"

def upload_documents(files):
    """Upload documents to the AutoMentor API for ingestion."""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file, file.type)))
        
        response = requests.post(
            INGEST_ENDPOINT,
            files=files_data,
            timeout=300
        )
        if response.status_code == 200:
            return True, response.json().get("message", "Documents ingested successfully!")
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {str(e)}\n\nMake sure the API server is running."

# Header
st.markdown('<div class="main-header">🚗 AutoMentor</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Your Local Automotive Knowledge AI Assistant</p>', unsafe_allow_html=True)

# Check API status
api_healthy = check_api_health()
st.session_state.api_status = api_healthy

# Sidebar
with st.sidebar:
    st.title("⚙️ Navigation")
    
    # API Status indicator
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Run: `uvicorn api.server:app --host 0.0.0.0 --port 8000`")
    
    st.markdown("---")
    
    # Navigation options
    page = st.radio(
        "Choose an option:",
        ["💬 Query Knowledge Base", "📁 Upload Documents"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Clear conversation button
    if page == "💬 Query Knowledge Base":
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    st.markdown("---")
    st.caption("AutoMentor v1.0")
    st.caption("Powered by Ollama + LangChain")

# Main content area
if page == "💬 Query Knowledge Base":
    st.header("💬 Chat with AutoMentor")
    
    if not api_healthy:
        st.warning("⚠️ The API server is not running. Please start it first.")
        st.code("uvicorn api.server:app --host 0.0.0.0 --port 8000", language="bash")
    else:
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history):
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])
        
        # Query input
        st.markdown("---")
        
        # Use chat input for better UX
        user_question = st.chat_input("Ask me anything about automotive knowledge...")
        
        if user_question:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Display user message immediately
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_question)
            
            # Show thinking spinner and get response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        response = send_query(user_question)
                    st.markdown(response)
            
            # Add assistant response to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
        
        # Show example queries if conversation is empty
        if len(st.session_state.conversation_history) == 0:
            st.markdown("### 💡 Example Questions:")
            examples = [
                "What is the battery capacity of Nissan Leaf 40kWh?",
                "How to troubleshoot engine misfire in Toyota Corolla?",
                "What is the recommended tire pressure for a 2020 Honda Civic?",
                "Explain the function of an EGR valve."
            ]
            
            cols = st.columns(2)
            for idx, example in enumerate(examples):
                with cols[idx % 2]:
                    if st.button(example, key=f"example_{idx}", use_container_width=True):
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": example
                        })
                        response = send_query(example)
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.rerun()

elif page == "📁 Upload Documents":
    st.header("📁 Upload Automotive Documents")
    
    if not api_healthy:
        st.warning("⚠️ The API server is not running. Please start it first.")
        st.code("uvicorn api.server:app --host 0.0.0.0 --port 8000", language="bash")
    else:
        st.markdown("""
        Upload your automotive documents (car manuals, specifications, guides) to expand AutoMentor's knowledge base.
        
        **Supported formats:** PDF, TXT, CSV
        """)
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            help="Select one or more documents to add to the knowledge base"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            for file in uploaded_files:
                st.text(f"• {file.name} ({file.size / 1024:.2f} KB)")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Upload and Ingest Documents", type="primary", use_container_width=True):
                    with st.spinner("Uploading and processing documents... This may take a few minutes."):
                        success, message = upload_documents(uploaded_files)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
        else:
            st.info("👆 Please select files to upload")
        
        st.markdown("---")
        st.markdown("### 📝 Instructions:")
        st.markdown("""
        1. Click **Browse files** to select documents
        2. Choose one or more PDF, TXT, or CSV files
        3. Click **Upload and Ingest Documents**
        4. Wait for processing to complete
        5. Start querying your new knowledge!
        
        ⚠️ **Note:** Ingestion can take several minutes depending on document size.
        """)
