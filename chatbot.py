from main import LolChatBot
import streamlit as st


st.set_page_config(page_title="Leaguage of Legend ChatBot")
with st.sidebar:
    st.title('Leaguage of Legend ChatBot')

@st.cache_resource
def get_bot():
    return LolChatBot()

bot = get_bot()
    

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain_with_source.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, can i help you today?"}]

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