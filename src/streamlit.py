import sys
sys.path.append(".")

import streamlit as st
from src import predict


# Title
st.title("HIV-1 recombination classifier")
st.write("\n")
st.write("This is an example of a HIV-1 recombination classifier using a \
small neural net to train codon (triplets/trigrams) based embeddings. This\
should not be used for other than demonstrating purposes. Please notice that\
the training dataset for this classifier doesn't include subtype B data.")

# Get run components for prediction
pred = 1

# Pages
page = st.sidebar.selectbox(
    "Choose a page", ['Classifier', 'Documentation'])
if page == 'Classifier':

    st.header("Enter genome to classify:")
    st.write("\n")
    st.write("How to use: \
        Just paste a DNA sequence on the following box and press RUN")
    # Input text
    text = st.text_input(
        "", value="atgagagtgatggggatcaagaggaactgtcaacaatggtggatatggggaatcttaggcttttggatgctaatgatttgtaatggaagggagaacatgtgggtcacagtctattatggggtacctgtgtggaaagaagcaaaaactactctattttgtgcatcagatgctaaagcatatgagaaagaagtgcataatgtctgggctacacatgcctgtgtacccacagaccccaacccacaagaaatggagttaaaaaatgtaacagaaaattttaacatgtggaaaaatgacatggtggatcaaatgcacgaggatataattagtttatgggatcaaagcctaaaaccatgtgtaaagttgaccccactctgtgtcactttaaactgtagtgctaccagcaatagtagtacttacaataatgtcacctacaatgagaccacaaaaggagacatgaaaaattgctctttcaatataaccacagaagtaagggataagaaaaagaaggaatatgcacttttttataggcttgatataacacctcttgatgagaaatccaatgacagtgagtatagattaataaattgtaatacctcagccataacacaagcctgtccaaaggtcacttttgacccaattcctatacattattgtactccagctggttatgcgattctaaagtgtaataataagacattcaatggaacaggaccatgcaataacgtcagcactgtacaatgtacacatggaattaagccagtggtatcaactcaactactgttaaacggtagtctagcagaagaagggataataattagatctgaaaatataacagacaatgtcaaaacaataatagtacatcttaatgaacctgtagaaattgtgtgtcaaaggcccggcaataacacaagacaaagtgtgaggataggaccaggacaaacattctatgcaacaggagacataataggagatataagagcagcacattgtaacattactgaagagcaatggaataaaactttaaacagggtaagagaaaaattaggagaatacttccctaatagaacaataaaatttgatcaacactcaggaggggacttagaaattacaacacatagctttaattgtagaggagaatttttctattgcaatacatcaaaattgttcacatacatgtggcctaacagtacaggagatacttcaaattcaaaaaacatcacaatccgatgcagaataagacaaattataaacatgtggcagggggtaggacgagcaatgtatgcccctcctgttgaagggaacataacatgtagatcaaatatcacaggactactattgacacgtgatggaggtaatggtaatgcagaaaatggctcagaaatattcagacctgcaggaggagatatgagggacaattggagaagtgaattatataaatataaagtgatagaaattaagccattaggactggcacccactaaggcaaaaaggcgagtggtggagagagaaaaaagagcagtgggaataggagctatgttccttgggttcttgggagtagcaggaagcactatgggcgcagcatcaataacgctgacggtacaggccagacaactgttgtctggtatagtgcaacagcaaagcaatttgctgaaggctatagaggcgcaacagcatctgttgcaactcacggtctggggcattaaacagctccaggcaagagtcctggctatggaaagatacctaaaggatcaacagctcctagggatttggggctgctctggaaaacgcatctgcaccactgccgtgccttggaacgccagttggagtaataaatcttacgagagaatttgggataacatgacatggatgcagtgggatagagaaattagtaactacacagacacaatatacaggttgcttgaagactcgcaaaaccagcaggaagaaaatgaaaaggagttactagaattggacagatggaacaatctgtggaattggtttggcataacaaactggctgtggtatataaaaatattcataatgatagtaggaggcttgataggtttaagaataatttttgctgtgctttctttagtaaatagagtcaggcagggatactcacctttgtcatttcagacccttaccccaaaccagaggggactcgacaggctcggaggaatcgaagaagaaggtggagagcaagacaaagacagatccattcgattagtgagcggattcttagcacttttctgggacgatctgaggagcctgtgccttttcagctaccaccgattgagagacttcatattggtgacagcgagagtggtggaacttctgggacgcagcagtctcaggggactacagaagggatgggcagcccttaagtatctgggaggtcttgtgcagtattgggggctagagctaaaaaagagtgctactagtctgcttgataccatagcaatagcagtagctgaaggaacagataggattatagaattagtacaaagaatttgtagagctatctaccacatacctacaagaataagacagggctttgaagcagctttgcaatag")

    # Run Button
    pressed = st.button('RUN')

    # Predict
    if pressed:
        results = predict.make_pred(text)

        # Results
        if results[0] == 0:
            outcome = 'No evidence of recombination'
        elif results[0] == 1:
            outcome = 'Evidence of recombination'
        else:
            outcome = 'Error'

        st.write("**Result**:", outcome)

    st.write("If you want to know more about this tool and how the result was  \
    produced please visit the Documentation page on the drop-down menu on \
    the left.")
    
elif page == 'Documentation':

    st.header("Preprocessing")
    st.write("""From the imputed genome the set of possible trigrams (ngrams of\
length 3) are created. Covering the 3 possible open reading frames. Each\
trigrams were then tokenized using keras. Since the sequenced region of the\
genome may have different lengths the tokenized inputs are then padded to\
length 3000. This data in then passed on the neural net showed in the next\
step. This net, containing an embedding layer, was trained on 103476 genomes\
(none B subtype) to classify HIV-1 genomes as recombinant or not.""")

    st.header("Classifier Neural Net Structure")
    st.markdown("![Alt Text](https://raw.githubusercontent.com/PMMAraujo/HIV-1_recombination_classifier/master/recombination_classifier/models_files/recomb_classifier_net.png)")


    st.header("Notes")
    st.write("The data set used can be found [here.](https://drive.google.com/file/d/1-Tim99TrSR8pGzLse3yN9sBBs08bFIlJ/view?usp=sharing)")
    st.write("The training and inference process can be consulted in the\
        notebook [here.](https://github.com/PMMAraujo/HIV-1_recombination_classifier/blob/master/model_creation_and_experimentation.ipynb)")
    st.write("The input genomes for this classifier don't need to be previously\
        aligned.")
    st.write("Since this is a toy example it should not be used for other\
        purposes than this demonstration.")
    st.write("Since this is a toy example information regarding the subtype B\
        was not included in the creation of this classifier.")
