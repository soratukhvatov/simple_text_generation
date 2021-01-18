
import urllib
import streamlit as st
import string
import torch
from transformers import pipeline
from pathlib import Path
import subprocess

list_files = subprocess.run(["ls", "-l"])
print("The exit code was: %d" % list_files.returncode)

list_files = subprocess.run(["ls", "mygpt2-medium", "-la"])
print("The exit code was: %d" % list_files.returncode)

from mytextgenerationpipeline import run_my_pplm
from transformers import GPT2Tokenizer

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    cloud_model_location = "1-CuaiNaxRDkm4m1rrego6Mwt-NJ7ZN7S"

    f_checkpoint = Path("mygpt2-medium/pytorch_model.bin")
    x = 0
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)

    list_files = subprocess.run(["ls", "mygpt2-medium", "-la"])
    print("The exit code was: %d" % list_files.returncode)


    # return pipeline("text-generation", model="mygpt2-medium")
    from transformers.modeling_gpt2 import GPT2LMHeadModel
    from transformers import AutoTokenizer

    # model = GPT2LMHeadModel.from_pretrained("mygpt2-medium")
    # tokenizer = AutoTokenizer.from_pretrained("mygpt2-medium")
    # >> > pipeline('ner', model=model, tokenizer=tokenizer)
    # load pretrained model
    pretrained = "mygpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(
        pretrained,
        output_hidden_states=True
    )
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained)

    return model, tokenizer#pipeline("text-generation", model="mygpt2-medium")#, tokenizer=tokenizer)#, config="mygpt2-medium")


model, tokenizer = load_model()
st.title('Text generation')
st.subheader(""" "The GPT-2 MEDIUM" model is fine-tuned on articles from Medium.com .\n
The following tags were selected: 
* Blockchain 
* Cryptocurrency 
* Polkadot 
* Sora""")
textbox = st.text_area('Start your story:', 'The original Blockchain is open-source technology which ',
                       height=200, max_chars=1000)
slider = st.slider('Max story length (in characters)', 50, 200, 140)
button = st.button('Generate')

listoftopics = ['legal', 'military', 'monsters', 'politics', 'religion', 'science', 'space', 'technology']
discriminators = ['clickbait', 'non clickbait', 'positive sentiment', 'neg sentiment']

topics = st.sidebar.radio("Select a Topic", listoftopics)
# step_size = st.sidebar.slider('Step size', 0.01, 0.1, 0.05)
# kl_scale = st.sidebar.slider('KL-scale', 0.0, 1.0)
# gm_scale = st.sidebar.slider('GM-scale', 0.0, 1.0, 0.95)
# num_iterations = st.sidebar.slider('Num iterations (impacts gen. time)', 1, 10, 2)
# gen_length = st.sidebar.slider('Gen. length (impacts gen. time)', 5, 20)
# is_sampling = st.sidebar.checkbox('Use sampling')

c1, c2 = st.sidebar.beta_columns(2)
rand_seed = c1.number_input('Random Seed', value=0)

# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
# x=0
# print(topics)
# 0/0
if button:
    # output_text = model(textbox, max_length=slider)[0]['generated_text']

    output_text = run_my_pplm(model, #"C:/Users/Admin/simple_text_generation/mygpt2-medium",
                                  tokenizer,
                                  cond_text=textbox,
                                  bag_of_words=topics,
                                  length=slider,
                                  uncond=False,
                                    verbosity='quiet',
                              colorama=True,
                              seed=rand_seed)

    # print(output_text)
    output_text = output_text[0][-1]

    # output_text = model(textbox, do_sample=True, max_length=slider, top_k=50, top_p=0.95, num_returned_sequences=1)[0][
    #     'generated_text']

    for i, line in enumerate(output_text.split("\n")):
        # if False and ":" in line:
        #     speaker, speech = line.split(':')
        #     st.markdown(f'__{speaker}__: {speech}')
        # else:
            line = ''.join([str(char) for char in line if char in string.printable])
            st.markdown(line, unsafe_allow_html=True)
# model(textbox, max_length=slider)[0]['generated_text']



# print(generated_texts)