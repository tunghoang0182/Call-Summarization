import streamlit as st
from openai import OpenAI
import json
import os



api_key = st.secrets["API_KEY"]

client = OpenAI(api_key=api_key)


if not os.path.exists('uploads'):
    os.makedirs('uploads')

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    
    return response



def summarize_text(text):
    summary_prompt = (
        f"You are a sales assistant responsible for summarizing a phone conversation between a customer and our sales representative at Sunwire Inc. "
        f"Your task is to extract and clearly present key information from the conversation that will be helpful for our sales team during future reviews. "
        f"Please ensure that key points from the conversation are concisely summarized. "
        f"Personal details such as names, addresses, and emails must be documented accurately with correct spelling. "
        f"Customer inquiries, requests, and any decisions made should be highlighted. "
        f"The sales representative should be identified as representing Sunwire Inc., and the other participant as the customer.\n\n"
        f"The summary should follow this format:\n\n"
        f"Summary:\n\n"
        f"Client Information:\n"
        f"(Include details such as name, address, phone number, and email, if mentioned.)\n\n"
        f"Phone Call Conversation Summary:\n"
        f"(Provide a brief but comprehensive summary of the key points discussed, focusing on customer inquiries, representative responses, and any outcomes.)\n\n"
        f"Customer Information:\n"
        f"(Identify the customer’s needs or interests based on their inquiries and responses during the call.)"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )
    return response.choices[0].message.content



# Streamlit UI
st.title("📝 Call Summarization")

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac", "webm"], label_visibility="collapsed")

if uploaded_file is not None:
    # Save the uploaded file to disk
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)  # Ensure the uploads directory exists
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    with st.spinner('Summarizing transcription...'):
        transcription_response = transcribe_audio(file_path)
        transcription_text = transcription_response.text  # Adjust based on your actual API response structure

        transcription_text_file = file_path.replace(os.path.splitext(file_path)[1], ".txt")
        with open(transcription_text_file, "w") as text_file:
            text_file.write(transcription_text)


        summary_text = summarize_text(transcription_text)
    
    
    st.markdown(summary_text)

    with open(transcription_text_file, "r") as text_file:
        st.download_button(
            label="Download Transcription",
            data=text_file.read(),
            file_name=os.path.basename(transcription_text_file),
            mime="text/plain"
        )


