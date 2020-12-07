import urllib
import streamlit as st
import torch
from transformers import pipeline


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    return pipeline("text-generation", model="test-clm")


model = load_model()

textbox = st.text_area('Start your story:', '', height=200, max_chars=1000)
slider = st.slider('Max story length (in characters)', 50, 200)
button = st.button('Generate')
x=0

if button:
    # output_text = mymodel(textbox, max_length=slider)[0]['generated_text']
    output_text = model(textbox, do_sample=True, max_length=slider, top_k=50, top_p=0.95, num_returned_sequences=1)[0][
        'generated_text']

    for i, line in enumerate(output_text.split("\n")):
        if ":" in line:
            speaker, speech = line.split(':')
            st.markdown(f'__{speaker}__: {speech}')
        else:
            st.markdown(line)
