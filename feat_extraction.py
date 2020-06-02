from typing import List
from nltk import word_tokenize
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from postagger import PosTagger
import re
import string
import numpy as np


def extract_bag_of_words(sentences: List[str]):
    stemmer = StemmerFactory().create_stemmer()
    tokens = set()
    splitted_sentence = [None]* len(sentences)
    for i, sentence in enumerate(sentences):
        data = word_tokenize(sentence.lower())
        data = (d for d in data if d not in string.punctuation)
        data = (stemmer.stem(d) for d in data)
        data = [d for d in data if len(d)>0]
        splitted_sentence[i] = data
        tokens.update(data)
    tokens = sorted(tokens)
    bag_of_words = [None]* len(sentences)
    for i, sentence in enumerate(splitted_sentence):
        counter = Counter(sentence)
        row = [(counter[word] if word in counter else 0) for word in tokens]
        bag_of_words[i] = row
    return bag_of_words, tokens

def extract_question_type(sentences: List[str]):
    stemmer = StemmerFactory().create_stemmer()
    label = ['wh_kenapa', 'wh_mana', 'wh_apa', 'wh_berapa', 'wh_siapa', 'wh_bagaimana', 'wh_kapan']
    label_list = [None]* len(sentences)
    for i, sentence in enumerate(sentences):
        row = [0, 0, 0, 0, 0, 0, 0]
        lower: str = sentence.lower()
        if ('kenapa' in lower or 'mengapa' in lower):
            row[0] = 1
        if (any(stemmer.stem(word)=='mana'  for word in word_tokenize(lower))):
            row[1] = 1
        if ('berapa' in lower):
            row[3] = 1
        if ('siapa' in lower):
            row[4] = 1
        if ('bagaimana' in lower):
            row[5] = 1
        if (any(re.match(r'^apa', word) for word in word_tokenize(lower))):
            row[2] = 1
        if ('kapan' in lower):
            row[6] = 1
        label_list[i] = row
    
    return label_list, label

def extract_word_shapes(sentences: List[str]):
    word_shapes = [
        ('ws_uppercase', lambda word: word.isupper() and word.isalpha()), 
        ('ws_lowercase', lambda word: word.islower() and word.isalpha()), 
        ('ws_mixedcase', lambda word: word.isalpha() and any(c.isupper() for c in word) and any(c.islower() for c in word)),
        ('ws_numeric', lambda word: re.match(r'[+-]?[0-9]+(\\.[0-9]+)?([Ee][+-]?[0-9]+)?', word)), 
        ('ws_other', lambda word: not any(func(word) for _, func in word_shapes[:-1]))]
    
    word_shape_list = [None] * len(sentences)
    for i, sentence in enumerate(sentences):
        tokenized = word_tokenize(sentence)
        tokenized = [word for word in tokenized if word not in string.punctuation]
        word_shape_freq = [0 for i in range(len(word_shapes))]
        for j, word_shape in enumerate(word_shapes):
            shape, func = word_shape
            word_shape_freq[j] = sum(1 for word in tokenized if func(word))
        
        word_shape_list[i] = word_shape_freq
    
    return word_shape_list, [x[0] for x in word_shapes]

def extract_postag(sentences):
    postagger = PosTagger()
    tag_name = set()
    tags = [None] * len(sentences)
    for i, sent in enumerate(sentences):
        tagged_sentence = postagger.tag(sent)
        tag_only = [tag for _, tag in tagged_sentence]
        tag_name.update(tag_only)
        tags[i] = tag_only
    tag_name = sorted(tag_name)
    tag_vector = [None] * len(sentences)
    for i, tag in enumerate(tags):
        counter = Counter(tag)
        row = [(counter[tm] if tm in counter else 0) for tm in tag_name]
        tag_vector[i] = row
    tag_name = [f"PT_{tag}"for tag in tag_name]
    return tag_vector, tag_name
    


def feature_extraction(sentences):
    list_feature_ext = [
        extract_bag_of_words,
        extract_question_type,
        extract_word_shapes,
        extract_postag
    ]
    result = [feat(sentences) for feat in list_feature_ext]
    return result

def feature_combination_train(fe_results) -> (np.ndarray, List[str]):
    label = list()
    collect = list()

    for value, name in fe_results:
        collect.append(value)
        label += name
    vector = np.concatenate(tuple(collect), axis=1)
    return vector, label

def feature_combination_test(fe_results, label_latih) -> np.ndarray:
    label = list()
    collect = list()

    for value, name in fe_results:
        collect.append(value)
        label += name
    vector = np.concatenate(tuple(collect), axis=1)
    new_vector = np.zeros((vector.shape[0], len(label_latih)))
    for i in range(len(label_latih)):
        lb = label_latih[i]
        if (lb in label):
            idx = label.index(lb)
            new_vector[:, i] = vector[:, idx]
    return new_vector


        
    
    

    
    

