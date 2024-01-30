import streamlit as st
import numpy as np
from keras.models import load_model
from Bio.Seq import Seq
import os

# Load the trained model
contigLengthk = "0.3"  # Assuming contig length is 1k for simplicity
filter_len1 = 10
nb_filter1 = 1000
nb_dense = 1000
nb_epoch = 30
accuracy = 0.89
model_path = f"models/model_siamese_varlen_{contigLengthk}k_fl{filter_len1}_fn{nb_filter1}_dn{nb_dense}_ep{nb_epoch}_acc{accuracy}.h5"
model = load_model(model_path)

# Function for encoding sequences into matrices of size 4 by n
def encodeSeq(seq):
    seq_code = []
    for pos in range(len(seq)):
        letter = seq[pos]
        if letter in ['A', 'a']:
            code = [1, 0, 0, 0]
        elif letter in ['C', 'c']:
            code = [0, 1, 0, 0]
        elif letter in ['G', 'g']:
            code = [0, 0, 1, 0]
        elif letter in ['T', 't']:
            code = [0, 0, 0, 1]
        else:
            code = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        seq_code.append(code)
    return seq_code

# Streamlit app
st.title("Virus Identification App")

# Input DNA sequence
sequence_input = st.text_area("Enter DNA sequence:", "", key="sequence_input")

# Make prediction when the user clicks the "Predict" button
if st.button("Predict"):
    if sequence_input:
        # Preprocess the input sequence
        seq = Seq(sequence_input)
        codefw = encodeSeq(seq)
        codebw = encodeSeq(seq.reverse_complement())

        # Make prediction using the loaded model
        score = model.predict([np.array([codefw]), np.array([codebw])], batch_size=1)
        # st.success(f"The predicted score for the input sequence is: {score[0][0]}")

        if score[0][0]<0.5:
            st.success(f"The dna Sequence contains No Virus")
        else:
            st.error(f"The dna Sequence contains Virus")


    else:
        st.warning("Please enter a DNA sequence for prediction.")




