import gradio as gr
from transformers import pipeline
import fitz  # PyMuPDF

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="t5-small", revision="main")

# Function to summarize text
def summarize_text(text, model):
    summary = model(text)[0]['summary_text']
    return summary

# Function to read PDF and summarize
def summarize_pdf(pdf_file, model):
    with fitz.open(pdf_file.name) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return summarize_text(text, model)

# Gradio Interface
def summarize(input_text, uploaded_file):
    if input_text:
        summary = summarize_text(input_text, summarizer)
    else:
        summary = summarize_pdf(uploaded_file, summarizer)
    return summary

inputs = [
    gr.Textbox(lines=10, label="Enter Text to Summarize"),
    gr.File(label="Upload PDF file")
]
output = gr.Textbox(label="Summary")

gr.Interface(
    fn=summarize,
    inputs=inputs,
    outputs=output,
    title="Text Summarization App",
    description="Summarize text or PDF files using pre-trained models."
).launch('share=True')
