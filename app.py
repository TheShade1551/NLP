# app.py
import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Set page configuration
st.set_page_config(
    page_title="Cross-Lingual Translator",
    page_icon="🌐",
    layout="wide"
)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return model, tokenizer

model, tokenizer = load_model()

# Supported languages with ISO codes
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl"
}

# Translation function
def translate(text, src_lang, tgt_lang):
    if not text.strip():
        return ""
    
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# UI Components
st.title("🌐 Cross-Lingual Translator")
st.markdown("Translate text between 100+ languages using AI")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    src_lang = st.selectbox("From language:", list(LANGUAGES.keys()), index=0)
    input_text = st.text_area("Enter text to translate:", height=200)
    
    # Language swap button
    if st.button("↔ Swap Languages"):
        st.session_state.src_lang, st.session_state.tgt_lang = st.session_state.tgt_lang, st.session_state.src_lang
        st.experimental_rerun()

with col2:
    st.subheader("Output")
    tgt_lang = st.selectbox("To language:", list(LANGUAGES.keys()), index=1, key="tgt_lang")
    output_text = st.text_area("Translation:", height=200, disabled=True)

# Handle translation
if st.button("Translate", type="primary"):
    if input_text.strip():
        with st.spinner("Translating..."):
            iso_src = LANGUAGES[src_lang]
            iso_tgt = LANGUAGES[tgt_lang]
            translation = translate(input_text, iso_src, iso_tgt)
            st.session_state.translation = translation
    else:
        st.warning("Please enter text to translate")

# Display translation
if 'translation' in st.session_state:
    output_text = st.session_state.translation
    st.experimental_rerun()

# Add footer
st.markdown("---")
st.caption("Powered by M2M100 model from Meta AI | Built with Streamlit")

# Initialize session variables
if 'src_lang' not in st.session_state:
    st.session_state.src_lang = src_lang
if 'tgt_lang' not in st.session_state:
    st.session_state.tgt_lang = tgt_lang
if 'translation' not in st.session_state:
    st.session_state.translation = ""
