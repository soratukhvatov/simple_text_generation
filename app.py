
import urllib
import streamlit as st
import torch
from transformers import pipeline
from pathlib import Path
import subprocess

list_files = subprocess.run(["ls", "-l"])
print("The exit code was: %d" % list_files.returncode)

list_files = subprocess.run(["ls", "mymodel","-la"])
print("The exit code was: %d" % list_files.returncode)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    
    cloud_model_location = "1-j9pfSXsS-o9BE3M2sjNWX8cI5r_TMsd"

    f_checkpoint = Path("mymodel/pytorch_model.bin")
    x = 0
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)

    list_files = subprocess.run(["ls", "mymodel","-la"])
    print("The exit code was: %d" % list_files.returncode)
    return pipeline("text-generation", model="mymodel")


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
