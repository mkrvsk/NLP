import gradio as gr
import cohere

def summarize_text(api_key, text):

    if not api_key or not text:
        return "Please provide both the API key and text."

    try:
        co = cohere.Client(api_key)

        response = co.generate(
            model='command',
            prompt=f"Create a concise, one-sentence summary of the following text:\n\n{text}\n\nSummary:",
            max_tokens=100,
            temperature=0.2,
            k=1,
            p=0.75
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Інтерфейс Gradio
description = """
# Text Summarization App  
This app uses Cohere's Command model to summarize text.  
Enter your Cohere API key and text to get a concise summary.
"""

with gr.Blocks() as demo:
    gr.Markdown(description)

    with gr.Row():
        api_key_input = gr.Textbox(label="Cohere API Key", placeholder="Enter your API Key here", type="password")
        text_input = gr.Textbox(label="Text to Summarize", placeholder="Enter the text you want to summarize", lines=5)

    with gr.Row():
        summarize_button = gr.Button("Summarize")
        result_output = gr.Textbox(label="Summary", lines=3)

    summarize_button.click(summarize_text, inputs=[api_key_input, text_input], outputs=result_output)

demo.launch()