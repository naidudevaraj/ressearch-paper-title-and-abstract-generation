# app.py
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer
from pdf2image import convert_from_bytes
import pytesseract
import torch
import re

# Configuration
ABSTRACT_MODEL = "sshleifer/distilbart-cnn-12-6"
TITLE_MODEL = "linydub/bart-large-samsum"
MAX_FILE_SIZE_MB = 10
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path!

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

@st.cache_resource
def load_models():
    """Load and cache models with proper tokenizers"""
    with st.spinner('🚀 Loading AI models (first time 2-5 mins)...'):
        # Abstract model
        abs_tokenizer = AutoTokenizer.from_pretrained(ABSTRACT_MODEL)
        abstractive = pipeline(
            "summarization",
            model=ABSTRACT_MODEL,
            tokenizer=abs_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # Title model
        title_tokenizer = AutoTokenizer.from_pretrained(TITLE_MODEL)
        title_pipe = pipeline(
            "text2text-generation",
            model=TITLE_MODEL,
            tokenizer=title_tokenizer,
            max_length=60
        )

    return abstractive, title_pipe, abs_tokenizer, title_tokenizer

def extract_text(pdf_file):
    """Handle both text and image-based PDFs"""
    try:
        # First try regular text extraction
        reader = PdfReader(pdf_file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        
        # Fallback to OCR if no text found
        if not text.strip():
            images = convert_from_bytes(pdf_file.getvalue())
            text = " ".join([pytesseract.image_to_string(img) for img in images])
            
        return clean_text(text)
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        return ""

def clean_text(text):
    """Remove headers/footers/section numbers"""
    patterns = [
        r'\n\s*(\d+)\s*\n',          # Page numbers
        r'Proceedings of .*?\n',      # Conference headers
        r'arXiv:\d+\.\d+v\d+.*?\n',   # arXiv footers
        r'©\d{4}.*?\n',               # Copyright
        r'http\S+',                   # URLs
        r'\b(?:Figure|Table)\s+\d+'   # Figure/table captions
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    return text.strip()

def generate_title(abstract, title_pipe):
    """Generate a concise and meaningful research paper title (4-5 words)."""
    prompt = f"Generate a short, research-style title (4-5 words) for this abstract: {abstract}"
    
    title = title_pipe(
        prompt,
        num_beams=5,
        early_stopping=True,
        max_length=10,  # Limit to ~4-5 words
        do_sample=False
    )[0]['generated_text'].strip()

    # Remove unwanted tokens
    title = title.replace("<pad>", "").replace("</s>", "").strip()

    # Ensure title is concise (4-5 words)
    words = title.split()
    if len(words) > 5:
        title = " ".join(words[:5])  # Keep only the first 5 words

    return title

def main():
    # Main title
    st.markdown("<h1 style='text-align: center;'>RESEARCH PAPER TITLE AND ABSTRACT GENERATION</h1>", 
                unsafe_allow_html=True)
    
    # Upload section
    col1, col2 = st.columns([4, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload here", type=["pdf"], label_visibility="collapsed")
    with col2:
        generate_btn = st.button("ENTER", use_container_width=True)

    if generate_btn and uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large! Max {MAX_FILE_SIZE_MB}MB allowed")
            return

        raw_text = extract_text(uploaded_file)
        if not raw_text.strip():
            st.warning("No text extracted - document might be corrupted")
            return

        abstract_pipe, title_pipe, abs_tokenizer, title_tokenizer = load_models()

        with st.status("Processing...", expanded=True) as status:
            try:
                # Processing steps
                st.write("📖 Analyzing document...")
                clean_abstract_text = raw_text[:2000]  # First 2000 characters
                
                st.write("✍️ Generating abstract...")
                abstract = abstract_pipe(
                    clean_abstract_text,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']

                st.write("🖋️ Creating title...")
                title = generate_title(abstract, title_pipe)

                status.update(label="Complete!", state="complete", expanded=False)

                # Display results
                st.markdown(f"""
                <div style='margin-top: 30px;'>
                    <p style='font-size: 14px; font-weight: bold;'>TITLE</p>
                    <p style='font-size: 14px; margin-bottom: 20px;'>{title}</p>
                    <p style='font-size: 12px; font-weight: bold;'>ABSTRACT</p>
                    <p style='font-size: 12px;'>{abstract}</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
