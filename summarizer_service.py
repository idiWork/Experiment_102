
import re
import nltk
import unicodedata
from gensim.summarization import summarize, keywords

def clean_and_parse_document(document):
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError("Document is not string or unicode.")
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

def summarize_text(text, summary_ratio = None, word_count = 30):
    sentences = clean_and_parse_document(text)
    cleaned_text = ' '.join(sentences)
    summary = summarize(cleaned_text, split = True, ratio = summary_ratio, word_count = word_count)
    return summary 

def init():  
    nltk.download('all')
    return

def run(input_str):
    try:
        return summarize_text(input_str)
    except Exception as e:
        return (str(e))
