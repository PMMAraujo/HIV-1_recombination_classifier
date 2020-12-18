# HIV-1 recombination classifier

This is the repository of a toy example for an HIV-1 recombination classifier. The corresponding web app can be found [here](https://hiv1-recombination-classifier.herokuapp.com/).

The classifier is a [neural net](https://raw.githubusercontent.com/PMMAraujo/HIV-1_recombination_classifier/master/src/models_files/recomb_classifier_net.png) with an embedding layer, that processes the HIV-1 genomes as triples (trigrams), representing the codons in every open reading frame.
The classifier was built using [tensorflow-keras](https://www.tensorflow.org/), the [NLTK](https://www.nltk.org/) package was used to obtain the trigrams. The web app interface was built on [Streamlit](https://www.streamlit.io/) and deployd on [Heroku](https://www.heroku.com/).

Web app: https://hiv1-recombination-classifier.herokuapp.com/

