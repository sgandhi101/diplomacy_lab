import spacy
from string import punctuation

# command line run: spacy download en_core_web_sm

nlp = spacy.load("en_core_web_lg")


def clean_up(text):
    # Tokenize --> to break text down into sentences, words, or other units

    doc = nlp(text)
    token_list = [token for token in doc]

    # Get rid of stop words --> like “if,” “but,” “or,” and so on
    filtered_tokens = [token for token in token_list if not token.is_stop]

    # Normalize words --> relates all forms of a word back to its simplest form, or lemma
    lemmas = [token for token in filtered_tokens if token.lemma_ not in punctuation]
    return lemmas



text_1 = ' "In this code, you set up some example text to tokenize, load spaCy’s English model, and then tokenize the text by passing it into the nlp constructor. This model includes a default processing pipeline that you can customize, as you’ll see later in the project section."'
hello = (clean_up(text_1))

text_2 = 'In this code, you set up some example text to tokenize, load spaCy’s English model, and'
hello_2 = (clean_up(text_2))
print(hello)

doc1 = nlp(text_1)
doc2 = nlp(text_2)
print(doc1.similarity(doc2))
