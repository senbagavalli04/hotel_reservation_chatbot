import gradio as gr
from huggingface_hub import InferenceClient


client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def hotel_booking_chat(message, history):
    messages = [{"role": "system", "content": "You are a hotel booking assistant. Help users with reservations, availability, pricing, and hotel details."}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(messages, max_tokens=512, stream=True, temperature=0.7, top_p=0.95):
        token = message.choices[0].delta.content
        response += token
        yield response


def preset_message(query):
    return query, None  



demo = gr.ChatInterface(hotel_booking_chat)

with demo:
    gr.Markdown("### 🔘 Quick Commands")
    gr.Button("Check room availability").click(preset_message, inputs=[], outputs=demo)
    gr.Button("Book a room").click(preset_message, inputs=[], outputs=demo)
    gr.Button("View hotel amenities").click(preset_message, inputs=[], outputs=demo)
    gr.Button("Cancel my reservation").click(preset_message, inputs=[], outputs=demo)
    gr.Button("Talk to customer support").click(preset_message, inputs=[], outputs=demo)

if __name__ == "__main__":
    demo.launch()
