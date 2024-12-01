from main import LolChatBot
import streamlit as st
import json
import os
from PIL import Image
import requests
from io import BytesIO

from prefect.blocks.system import JSON

# Cấu hình trang
st.set_page_config(page_title="League of Legends ChatBot", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 1000px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load patch data function
@st.cache_resource
def get_bot():
    return LolChatBot()

bot = get_bot()

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain_with_source.invoke(input)
    return result


@st.cache_data
def load_patch_data():
    try:
        json_block = JSON.load("lol-latest-patch-info")
        return json_block.value
    except FileNotFoundError:
        return []


# Hàm hiển thị thông tin patch trong sidebar
def display_patch_info(patch):
    st.sidebar.title("League of Legends Patch Update Chatbot")
    st.sidebar.header(f"{patch.get('title', 'N/A')}")
    st.sidebar.write(f'Updated on: {patch.get("time", "N/A")}')
    
    # Hiển thị hình ảnh overview
    if "overview_image" in patch:
        try:
            custom_css = '''
                <style>
                    button[title="View fullscreen"]{
                        visibility: visible;
                        position: absolute;
                        top: 5px;
                        right: 5px;
                        z-index: 999;
                        border-radius: 4px;
                        padding: 4px;
                        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.2);
                    }
                    div[data-testid="stSidebar"] img {
                        position: relative;
                    }
                </style>
            '''
            st.markdown(custom_css, unsafe_allow_html=True)

            st.sidebar.image(
                patch["overview_image"],
                caption="Patch Overview",
                use_column_width=True, # Manually Adjust the width of the image as per requirement
            )
        except:
            st.sidebar.write("Could not load the image.")

    # Hiển thị các thông tin patch còn lại    
    description = patch.get("description", "N/A")
    st.sidebar.markdown(f'<div style="text-align: center"> {description} </div>', unsafe_allow_html=True)

    if "url" in patch:
        st.sidebar.write("")
        st.sidebar.markdown(f'See details on Riot page: [click here]({patch["url"]})')

# Load patch data
patch_data = load_patch_data()

# Hiển thị thông tin patch trong sidebar
if patch_data:
    display_patch_info(patch_data)  
else:
    st.sidebar.write("No patch data available.")


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! You can ask me about this latest patch update for League of Legends. I'm here to provide details on the newest changes, improvements, and updates to keep you informed."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
input = st.chat_input("Type your message here...")
if input:
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Create a placeholder for the assistant's message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for the spinner
        with message_placeholder.container():
            with st.spinner("Chatbot is thinking..."):
                response = generate_response(input)
        
        # Update the placeholder with the actual response
        message_placeholder.write(response)  # Replace spinner with the response directly
    
    # Append the response to session_state
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
