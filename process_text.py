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


text_1 = 'The head of the European Commission said the bloc would switch policy, under certain conditions, after more than a year of mostly banning nonessential travel.'
hello = (clean_up(text_1))

text_2 = 'American tourists who have been fully vaccinated against Covid-19 will be able to visit the European Union over the summer, the head of the bloc’s executive body said'
hello_2 = (clean_up(text_2))
print(hello)

doc1 = nlp(text_1)
doc2 = nlp(text_2)
print(doc1.similarity(doc2))
