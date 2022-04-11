import os
import re
import json
import nltk
import spacy
import random
import inflect
import numpy as np
from string import punctuation
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
num2word = inflect.engine()

punctuation = punctuation.replace("'", "")
punctuation = punctuation.replace("-", "")
punctuation += '¡«†â€ž'

def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(text):
    doc = nlp(text)
    text = []
    for token in doc:
        if token.lemma_ == '-PRON-':
            text.append(str(token))
        else:
            text.append(str(token.lemma_))
    text = ' '.join(text)
    return text

def replace_numbers(text):
    text = nltk.word_tokenize(text)
    result = []
    for i in text:
        if i.isnumeric():
            result.append(num2word.number_to_words(int(i)))
        else:
            result.append(i)
    result = ' '.join(result)
    return result

def expand_contractions(text):
    text = text.replace("can't","cannot")
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def clean_doc(doc):

    doc = doc.lower()
    
    doc = doc.replace("\n", "")
    doc = doc.replace("\x00", "c")
    doc = doc.replace("  ", " ")
    doc = doc.replace("-", " ")
    doc = doc.replace("xxx ", "")
    doc = doc.replace("0ceipt", "receipt")

    doc = doc.translate(str.maketrans("","", punctuation))

    doc = replace_numbers(doc)
    doc = expand_contractions(doc)
    doc = lemmatize(doc)

    return doc

def remove_preceding(txt, char):
    char_ind = txt.find(char) + len(char)
    txt = txt[char_ind:]
    return txt

def extract_cslu():
    loc = 'raw_data\\cslu_kids_raw\\trans'

    grade_transcripts = {}
    subdocs1 = os.listdir(loc)

    z_vals = [' z ','>z','z<'," z*"," z\n"," zed ", " p <", "and g ", "v  <pau>", " s<sing>", " f<bn>", "<ln> u v"]

    for i in subdocs1:
        grade_transcripts[i] = []
        subloc1 = os.path.join(loc, i)
        subdocs2 = os.listdir(subloc1)
        for j in subdocs2:
            subloc2 = os.path.join(subloc1, j)
            subdocs3 = os.listdir(subloc2)
            for k in subdocs3:
                subloc3 = os.path.join(subloc2, k)
                subdocs4 = os.listdir(subloc3)
                for l in subdocs4:
                    doc_name = os.path.join(subloc3, l)
                    with open(doc_name, 'r') as file:
                        doc = file.read()
                        for m in z_vals:
                            z_ind = doc.find(m)
                            if z_ind != -1:
                                if "<" in m:
                                    z_ind = z_ind + len(m) - 1
                                else:
                                    z_ind = z_ind + len(m)
                                break
                        if z_ind != -1:
                            doc = doc[z_ind:]
                        doc = re.sub("[\<\[].*?[\>\]]", "", doc)
                        doc = clean_doc(doc)
                        grade_transcripts[i].append(doc)    

    return grade_transcripts

def extract_treebank(corpus):
    loc = os.path.join('raw_data', corpus)
    docs = os.listdir(loc)
    grade_transcripts = {}
    for i in docs:
        doc_name = os.path.join(loc, i)
        with open(doc_name, 'r') as file:
            doc = file.readlines()

        age = list(np.where(np.array([i[:3] for i in doc]) == "@ID")[0])[0]
        age = doc[age].split('|')[3].split(';')[0]
        if len(age) == 0:
            continue

        grade = '0' + str(int(age) - 5)
        if grade not in grade_transcripts.keys():
            grade_transcripts[grade] = []
        
        transcript = []
        for line in doc:
            if line[1:4] == 'CHI':
                transcript.append(line[6:])
        
        if len(transcript) == 0:
            print('{}: no transcript'.format(doc_name))
        
        doc = ' '.join(transcript)
        doc = clean_doc(doc)
        grade_transcripts[grade].append(doc)
    
    return grade_transcripts

def extract_sb():
    sb_conventions = ['\n','\t','...','..','(0)','( )','(H)','\x7f','Hx','VOX','SM','YWN','TSK','THROAT','PAR','SNAP','<HI','HI>','SWALLOW','SNIFF','MUSIC_BECOMES_AUDIBLE','KISS','LATERAL_CLICK','LAUGHING','SEC','OVERHEAD_LIGHT_GOES_ON_BY_ITSELF','SLAPPING','MUSIC_STARTS','MUSIC_STOPS','WHISTLE','SNEEZE','ALAN','JON','MRC','MICROPHONE','POUND','J,','_M,','_P_','CLICK','P ','Q ','X','Y ','1','2','3','4','5','6','7','8','9','0']

    loc = 'raw_data\\santa_barbara_raw'
    doc_names = os.listdir(loc)

    docs = []
    for doc_name in doc_names:
        doc_name = os.path.join(loc, doc_name)
        with open(doc_name, 'r') as file:
            doc = file.readlines()
            for i in range(len(doc)):
                doc[i] = remove_preceding(doc[i], '\t')
                doc[i] = remove_preceding(doc[i], '\t')
            doc = ' '.join(doc)
            for i in sb_conventions: doc = doc.replace(i, "")
            doc = clean_doc(doc)
            docs.append(doc)

    docs = ' '.join(docs)

    unique_words = []
    for i in nltk.word_tokenize(docs):
        if i not in unique_words:
            unique_words.append(i)

    result = {'corpus':docs, 'unique_words':unique_words}

    return result

def shorten_corpus(orig_corpora, limit):

    corpora = orig_corpora.copy()
    corpora.pop('full')

    for i in corpora:
        corpus = corpora[i]
        if len(corpus) >= limit:
            random.Random(0).shuffle(corpora[i])
            corpora[i] = corpora[i][:limit]

    full_corpus = []
    for i in corpora:
        for j in corpora[i]:
            full_corpus.append(j)
    
    corpora['full'] = full_corpus

    return corpora

def write_docs():
    cslu_corpus = extract_cslu()

    print('CSLU Complete')
    
    gillam_corpus = extract_treebank('gillam')

    print('Gillam Complete')
    
    enni_corpus = extract_treebank('enni')
    
    print('ENNI Complete')
    
    pm_corpus = extract_treebank('peterson_mccabe')

    print('Peterson-McCabe Complete')    

    sb_corpus = extract_sb()

    print('Santa Barbara Complete')

    kids_corpora = {}
    for i in [cslu_corpus, gillam_corpus, enni_corpus, pm_corpus]:
        for j in i:
            if j not in kids_corpora:
                kids_corpora[j] = []
            kids_corpora[j].extend(i[j])
    
    kids_corpora = {i:kids_corpora[i] for i in sorted(kids_corpora)}
    full_corpus = []
    for i in kids_corpora:
        for j in kids_corpora[i]:
            full_corpus.append(j)
    
    kids_corpora['full'] = full_corpus

    kids_corpora_short = shorten_corpus(kids_corpora, limit=193)

    with open('corpora_kids.json', 'w') as file:
        json.dump(kids_corpora, file)

    with open('corpora_short.json', 'w') as file:
        json.dump(kids_corpora_short, file)

    with open('corpora_adult.json', 'w') as file:
        json.dump(sb_corpus, file)
