import streamlit as st
from openai import OpenAI
import json
import os



with open('api_key.json', 'r') as key_file:
    api_key = json.load(key_file)['key']

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
        f"Phone Call Conversation: {text}\n"
        "You are an AI sales assistant tasked with summarizing a phone call conversation between a customer and our sales representative. Your primary goal is to extract and organize information provided by the customer only. Do not include any information or statements made by the sales representative.\n\n"
        "Contact Details\n"
        "First Name:\n"
        "Last Name:\n"
        "Position:\n"
        "Phone (Main):\n"
        "Phone (Other):\n"
        "Fax Number:\n"
        "Email:\n\n"
        "Company Details\n"
        "Company:\n"
        "Acronym:\n"
        "Address Number:\n"
        "Address Street:\n"
        "Address Other:\n"
        "Unit:\n"
        "City:\n"
        "Province / State:\n"
        "Postal/Zip Code:\n"
        "Country:\n\n"
        "Service Package\n"
        "Selected Package:\n"
        "Monthly Cost:\n"
        "Promotion:\n"
        "Number of Lines:\n"
        "Features:\n\n"
        "Client Note\n"
        "(Notes about the client's needs and considerations)\n\n"
        "Installation Details\n"
        "Scheduled Date:\n"
        "Time Window:\n"
        "Technician:\n\n"
        "Follow-Up Plan\n"
        "Send Proposal:\n"
        "Follow-Up Call:\n"
        "Purpose:\n\n"
        "Timeline\n"
        "Today:\n"
        "[Date]:\n\n"
        "Question From the Customer:\n"
        "(List only the actual questions asked by the customer during the conversation. Do not infer or create questions.)\n\n"
        "Remember: Only include information explicitly provided by the customer. Do not add any details from the sales representative or make any assumptions about missing information."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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

def format_summary(summary_text):
    sections = summary_text.split('\n\n')
    formatted_sections = []
    for section in sections:
        lines = section.split('\n')
        if len(lines) > 1:
            formatted_section = f"## {lines[0]}\n"
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    formatted_section += f"**{key.strip()}:** {value.strip()}\n\n"  # Add an extra newline here
                else:
                    formatted_section += f"{line}\n\n"  # Add an extra newline for non key-value lines too
        else:
            formatted_section = f"{section}\n\n"  # Add newlines to single-line sections as well
        formatted_sections.append(formatted_section)
    return '\n'.join(formatted_sections)  # Join with single newline as we've added extra newlines in the formatting



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
        summary_text = summarize_text(transcription_text)
    
    formatted_summary = format_summary(summary_text)
    st.markdown(formatted_summary)
