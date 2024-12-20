import os
import json
import datetime
import requests
import threading

import wandb
import streamlit as st
from PIL import Image
from prefect.blocks.system import JSON

from main import LolChatBot


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

class PatchDataLoader:
    def __init__(self):
        self.current_patch_version = None
    
    def load_and_check(self):
        """Load patch data and check if it's new"""
        try:
            json_block = JSON.load("lol-latest-patch-info")
            new_data = json_block.value
            
            needs_refresh = False
            if self.current_patch_version is not None:
                if new_data.get("version", "") != self.current_patch_version:
                    needs_refresh = True
                    
            self.current_patch_version = new_data.get("version", "")
            return new_data, needs_refresh
            
        except ValueError:
            return {"temp": "temp"}, False
    
    @staticmethod
    def refresh_client():
        """Force client reload and clear cache/session"""
        st.info("New patch detected! Refreshing data...", icon="🔄")
        time.sleep(1)  # Give user time to see the message

        st.cache_data.clear()
        
        # Clear session state except for essential items
        keys_to_keep = []  # Add any keys you want to preserve
        for key in st.session_state.keys():
            if key not in keys_to_keep:
                del st.session_state[key]
        
        # Force client reload using JavaScript
        st.rerun()

# Create loader instance
@st.cache_resource
def get_loader():
    return PatchDataLoader()

@st.cache_resource
def get_bot():
    return LolChatBot()

# create bot
bot = get_bot()

# Load data with daily TTL
@st.cache_data(ttl=datetime.timedelta(days=1))
def load_patch_data():
    loader = get_loader()
    data, needs_refresh = loader.load_and_check()
    
    if needs_refresh:
        loader.refresh_client()
        
    return data


def display_patch_info(patch):
    st.sidebar.title("League of Legends Patch Update Chatbot")
    st.sidebar.header(f"{patch.get('title', 'N/A')}")
    st.sidebar.write(f'Updated on: {patch.get("time", "N/A")}')
    
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
  
    description = patch.get("description", "N/A")
    st.sidebar.markdown(f'<div style="text-align: center"> {description} </div>', unsafe_allow_html=True)

    if "url" in patch:
        st.sidebar.write("")
        st.sidebar.markdown(f'See details on Riot page: [click here]({patch["url"]})')


def generate_response(query):
    response = bot.predict_stream(query)

    return response


def start_log_feedback(feedback):
    print("Logging feedback.")
    st.session_state.feedback_given = True
    st.session_state.sentiment = feedback
    thread = threading.Thread(target=log_feedback, args=(st.session_state.sentiment,
                                                         st.session_state.query,
                                                         st.session_state.constructed_query,
                                                         st.session_state.context,
                                                         st.session_state.response))
    thread.start()
    if st.session_state.sentiment == "positive":
        st.toast(body="Thanks for the positive feedback!", icon="🔥")
    else:
        st.toast(body="Thanks for the feedback. We'll try to improve!", icon="😔")


def log_feedback(sentiment, query, constructed_query, context, response):
    ct = datetime.datetime.now()
    wandb.init(project="LoL-ChatBot",
               name=f"query: {ct}")
    
    formatted_context = json.dumps(context, indent=2) # context is an array of dict ([{content: xxx, metadata: yyy}])
    formatted_constructed_query = json.dumps(constructed_query, indent=2)
    
    table = wandb.Table(columns=["sentiment", "query", "constructed_query", "context", "response"])
    table.add_data(sentiment,
                   query,
                   formatted_constructed_query,
                   formatted_context,
                   response
                   )
    wandb.log({"Query Log": table})
    wandb.finish()


# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'constructed_query' not in st.session_state:
    st.session_state.constructed_query = False
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# Load patch data
patch_data = load_patch_data()
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
                response = st.write_stream(generate_response(input))
    
    # assign state
    st.session_state.query = input
    st.session_state.constructed_query = bot.constructed_query
    st.session_state.context = bot.context
    st.session_state.response = response
    st.session_state.sentiment = None
    st.session_state.feedback_given = False
    # Append the response to session_state
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

# display feedback button
if st.session_state.response and not st.session_state.feedback_given:
    col1, col2 = st.columns([1, 15])
    with col1:
        st.button('👍', key='positive_feedback', disabled=False, on_click=start_log_feedback, args=["positive"])

    with col2:
        st.button('👎', key='negative_feedback', disabled=False, on_click=start_log_feedback, args=["negative"])
