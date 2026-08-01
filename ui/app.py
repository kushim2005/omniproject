import streamlit as st
import requests
import time
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="OmniBrain UI", page_icon="🧠", layout="wide")

st.title("🧠 OmniBrain: Agentic Multi-Modal RAG")
st.markdown("Upload a PDF to ingest, then ask questions about text and embedded charts.")

# ── Sidebar: Document Upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Uploading and starting ingestion..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 202:
                        data = response.json()
                        job_id = data["job_id"]
                        st.success(f"Upload accepted! Job ID: {job_id}")
                        
                        # Poll status
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            status_res = requests.get(f"{API_URL}/upload/status/{job_id}")
                            if status_res.status_code == 200:
                                status_data = status_res.json()
                                state = status_data["status"]
                                msg = status_data.get("message", state)
                                
                                status_text.text(f"Status: {state.upper()} - {msg}")
                                
                                if state == "completed":
                                    progress_bar.progress(100)
                                    st.success(f"Ingestion complete! Text chunks: {status_data.get('text_chunks_upserted')}, Images: {status_data.get('images_upserted')}")
                                    break
                                elif state == "failed":
                                    st.error(f"Ingestion failed: {status_data.get('error')}")
                                    break
                                else:
                                    progress_bar.progress(50) # Indeterminate progress
                            time.sleep(2)
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

# ── Main Area: Chat Interface ────────────────────────────────────────────────
st.header("2. Ask OmniBrain")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_path in message["images"]:
                # Ensure the path is correct or load from API if needed.
                # Since UI might run in a separate container, it needs access to the image.
                # Assuming images are accessible locally for this MVP or served via API.
                # For now, we'll try to display local path.
                try:
                    st.image(img_path, caption=os.path.basename(img_path), width=400)
                except Exception as e:
                    st.warning(f"Could not load image {img_path}")

# React to user input
if prompt := st.chat_input("What would you like to know about the document?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare request
    payload = {
        "query": prompt,
        "top_k": 3,
        "search_text": True,
        "search_images": True
    }

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Simulated Thought Process Expander for LangGraph Agents
        with st.status("Agentic Thought Process...", expanded=True) as status:
            st.write("🕵️‍♂️ **Supervisor Agent:** Analyzing query intent...")
            time.sleep(1)
            st.write("🔍 **Search Agent:** Retrieving relevant text chunks from Qdrant...")
            time.sleep(1)
            st.write("🖼️ **Vision Agent:** Searching for visually similar charts/tables...")
            
            try:
                # Call actual Backend API
                response = requests.post(f"{API_URL}/query", json=payload)
                response.raise_for_status()
                data = response.json()
                
                text_results = data.get("text_results", [])
                image_results = data.get("image_results", [])
                
                st.write(f"✅ Found {len(text_results)} text chunks and {len(image_results)} images.")
                status.update(label="Query Processed Successfully", state="complete", expanded=False)
                
                # Formulate final response
                final_answer = f"Here is what I found based on your query: **{prompt}**\n\n"
                
                for idx, t in enumerate(text_results, 1):
                    final_answer += f"**Source {idx} (Page {t['page']}):**\n> {t['text']}...\n\n"
                
                st.markdown(final_answer)
                
                image_paths = []
                if image_results:
                    st.markdown("### Relevant Visual Context")
                    for img in image_results:
                        img_path = img['image_path']
                        # To display images from backend volume, UI needs volume mount or API endpoint.
                        # Assuming they share the /app/uploads volume via docker-compose
                        try:
                            st.image(img_path, caption=f"Page {img['page']}", width=400)
                            image_paths.append(img_path)
                        except Exception as e:
                            st.error(f"Image not found at {img_path}. (Ensure UI container mounts the uploads volume)")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "images": image_paths
                })
                
            except Exception as e:
                status.update(label="Error in Query Pipeline", state="error", expanded=True)
                st.error(f"Failed to fetch results: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
