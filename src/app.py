"""Streamlit interface: type a short text, get a predicted category.

Run:  streamlit run src/app.py
"""
import streamlit as st

from predict import predict

st.set_page_config(page_title="Smart Text Classifier", page_icon="🗂️")
st.title("Smart Text Classifier")
st.caption("Classifies short text as Complaint, Inquiry, Feedback or Other "
           "(TF-IDF + Logistic Regression).")

EXAMPLES = [
    "My parcel arrived damaged and nobody is replying to me",
    "What are your delivery charges to Lahore?",
    "Love the new design, checkout feels much faster",
    "Good morning",
]

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["text"] = ex

text = st.text_area("Enter text", key="text", height=120,
                    placeholder="e.g. I was charged twice for the same payment")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        result = predict(text)
        st.subheader(f"Prediction: {result['label']}")
        st.write(f"Confidence: **{result['confidence']:.1%}**")
        if result["confidence"] < 0.5:
            st.info("Low confidence - the text may be ambiguous or mix several intents.")
        st.bar_chart(result["probabilities"])
