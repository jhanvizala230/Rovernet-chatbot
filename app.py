import streamlit as st
import numpy as np
from PIL import Image
import os
from backend.main import predict  # Make sure this import matches your project structure
import os
os.environ["TORCH_DISABLE_NVFUSER"] = "1"

st.set_page_config(layout="wide")
st.title("🚀 Rovernet Chatbot")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "processed_image_path" not in st.session_state:
    st.session_state.processed_image_path = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# Layout
chat_col, image_col = st.columns([3, 1])

# Right: Show images
with image_col:
    st.markdown("### 🖼️ Terrain Result")
    if st.session_state.processed_image_path and os.path.exists(st.session_state.processed_image_path):
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width =True)
        st.image(st.session_state.processed_image_path, caption="Detection Results", use_container_width =True)
    elif st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width =True)
    else:
        st.info("Upload a Mars terrain image to begin.")

# Left: Chat area
with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Upload file
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None and uploaded_file.name != st.session_state.get("filename"):
        try:
            img = Image.open(uploaded_file).convert("RGB")

            # Reset session state for new image
            st.session_state.uploaded_image = img
            st.session_state.filename = uploaded_file.name
            st.session_state.processed_image_path = None
            st.session_state.messages = []  # Clear chat messages for new session

            st.success(f"✅ Image '{uploaded_file.name}' uploaded successfully.")
            st.rerun()  # Refresh to update all state-dependent views
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")

    # Chat input
    prompt = st.chat_input("Ask a question about the terrain...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            full_response = ""
            response_placeholder = st.empty()

            if st.session_state.uploaded_image:
                image_array = np.array(st.session_state.uploaded_image.convert("RGB"))

                # Call your backend
                response_stream, output_path = predict(
                    image_array,
                    st.session_state.filename,
                    prompt,
                    stream=True
                )
             

                for chunk in response_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                   # Show processed image
                if os.path.exists(output_path):
                    st.session_state.processed_image_path = output_path
                    # st.rerun()

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()


            else:
                fallback = "Please upload an image first."
                response_placeholder.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})
