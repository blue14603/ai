import streamlit as st
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set up the webpage title and layout
st.set_page_config(page_title="Qwen2.5-VL OCR App", layout="centered")
st.title("📄 Qwen2.5-VL Simple OCR UI")
st.write("Upload an image to extract text using Alibaba's Qwen2.5-VL-3B model.")

# Use caching so the model only loads ONCE into memory, not on every click
@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Automatically detects CUDA if available, otherwise falls back to CPU
    device = "mps" if torch.cuda.is_available() else "cpu"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device

# Show a loading spinner while the model loads
with st.spinner("Loading Qwen2.5-VL-3B into memory... (This may take a minute the first time)"):
    model, processor, device = load_model()

st.success(device.upper() + " Hardware Detected & Model Loaded Successfully!")

# File uploader widget
uploaded_file = st.file_uploader("Choose a document image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Prompt option (let user customize what they want)
    user_prompt = st.text_input(
        "Instruction for the AI:",
        placeholder="Type your query here...",
        key="user_query"
    )
    
    # Trigger button for OCR processing
    if st.button("Extract Text", type="primary"):
        with st.spinner("Processing image and extracting text..."):
            try:
                # Prepare conversation template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                
                # Process inputs
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(device)
                
                # Generate text response
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                # Display results
                st.subheader("📝 Extracted Result:")
                st.text_area(label="Result text (you can copy this)", value=output_text, height=250)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
