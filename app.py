import os

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from st_audiorec import st_audiorec
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openapi_key = os.getenv("OPENAI_API_KEY")




# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=openapi_key,
    max_tokens=100
)

# Set Streamlit page configuration
st.set_page_config(page_title="Voice-to-Voice Chatbot")
st.title("Voice-to-Voice Chatbot ")

github_link = "[GitHub Profile](https://github.com/abdullah0325)"
linkedin_link = "[LinkedIn Profile](https://www.linkedin.com/in/muhammad-abdullah-41b82028b/)"
facebook_link = "[Facebook Profile](https://www.facebook.com/profile.php?id=100090638882853)"

# Display social media profile links in the sidebar
st.sidebar.title("Muhammad Abdullah")
st.sidebar.write(github_link, unsafe_allow_html=True)
st.sidebar.write(linkedin_link, unsafe_allow_html=True)
st.sidebar.write(facebook_link, unsafe_allow_html=True)

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses
if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""  # Store the latest transcript

def transcribe_audio(file):
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        response_format="text"
    )
    return transcript

def build_message_list():
    """Build a list of messages including system, human, and AI chatbot messages."""
    messages = [SystemMessage(content="You are a knowledgeable AI assistant. Answer the user's questions  as accurately as possible.maximum answers should be in those language in which the will be asked")]
    for human_msg, ai_msg in zip(st.session_state['past'], st.session_state['generated']):
        if human_msg:
            messages.append(HumanMessage(content=human_msg))
        if ai_msg:
            messages.append(AIMessage(content=ai_msg))
    return messages

def generate_response(user_query):
    """Generate AI response using the ChatOpenAI model."""
    messages = build_message_list()
    messages.append(HumanMessage(content=user_query))
    ai_response = chat(messages)
    return ai_response.content



def text_to_speech(response):
    client = OpenAI(api_key=openapi_key)
    tts_response = client.audio.speech.create(model="tts-1", voice="nova", input=response)
    audio_content = tts_response.content  # Get audio content as bytes
    return audio_content

# Streamlit UI for audio recording and transcription
st.header("Voice-to-Voice Chat in Urdu")

audio_data = st_audiorec()
if audio_data is not None:
    st.audio(audio_data, format='audio/wav')
    audio_file = "recorded_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_data)
    with st.spinner("Transcribing..."):
        with open(audio_file, "rb") as file:
            st.session_state.transcript = transcribe_audio(file)

# Chatbot interaction
if st.session_state.transcript:
    user_query = st.session_state.transcript
    st.session_state.past.append(user_query)
    output = generate_response(user_query)
    st.session_state.generated.append(output)
    
    # Convert AI response to speech
    tts_file = text_to_speech(output)
    st.audio(tts_file, format='audio/mp3')

# Display chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.write("AI: ", st.session_state["generated"][i])
        st.write("User: ", st.session_state['past'][i])
