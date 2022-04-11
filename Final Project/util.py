import sys
import json
import nltk
import numpy as np
import pandas as pd

def load_data(file_name):
    # Loads all required packages and data
    sys.stdout.write(f'Loading {file_name}')
    if file_name == 'corpora_adult':
        with open('data/corpora_adult.json', 'r') as file:
            data = json.load(file)
    elif file_name == 'corpora_kids':
        with open('data/corpora_kids.json', 'r') as file:
            data = json.load(file)
    elif file_name == 'fdist':
        with open('data/fdist.json', 'r') as file:
            data = json.load(file)
    elif 'context_model' in file_name:
        try:
            with open(f'data/{file_name}.json', 'r') as file:
                data = json.load(file)
        except:
            sys.stdout.write('\r')    
            sys.stdout.write(f'Loading {file_name} ...not found\n')
            return
    elif 'word_degrees' in file_name:
        try:
            with open(f'data/{file_name}.json', 'r') as file:
                data = json.load(file)
        except:
            sys.stdout.write('\r')    
            sys.stdout.write(f'Loading {file_name} ...not found\n')
            return
    elif file_name == 'res':
        with open('data/res.json', 'r') as file:
            remove_words = json.load(file)
        data = []
        for i in remove_words:
            for j in remove_words[i]:
                data.append(j)
    else:
        print('no file identified')
        return
    sys.stdout.write('\r')    
    sys.stdout.write(f'Loading {file_name} ...done\n')
    return data

def p_print(i, n, word=None):
    # Displays a percentage complete bar with the word optionally added on
    len_bar = n / 30
    p = n / 100
    sys.stdout.write('\r')
    if word is None:
        sys.stdout.write("[%-30s] %d%% complete" % ('='*int((i+1)/len_bar), (i+1)/p))
    else:
        word = word + (" " * 20)
        sys.stdout.write("[%-30s] %d%% complete - Computing word %d/%d: %s" % ('='*int((i+1)/len_bar), (i+1)/p, i+1, n, word))

def unique(vec):
    # Finds unique objects in a list
    result = []
    for i in vec:
        if i not in result:
            result.append(i)
    return result

def cosine_similarity(v1, v2):
    # Computes cosine similarity between two vectors
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def most_similar(word, kids_ppmi, adult_ppmi, top_n):
    # Finds the most similar vector in a matrix of vectors
    similarities = [cosine_similarity(adult_ppmi[i], kids_ppmi[word]) for i in kids_ppmi]
    top_words = similarities.copy()
    top_words.sort(reverse=True)
    top_words = top_words[:top_n]
    top_ind = [int(np.where(np.array(similarities) == i)[0][0]) for i in top_words]
    top_words = {list(kids_ppmi)[i]: similarities[i] for i in top_ind}
    return top_words

def gen_z_score(vec):
    # Scales a vector to the z-distribution
    z_score = [(i - np.mean(vec)) / np.std(vec) for i in vec]
    return z_score

def gen_fdist(corpora, grades, save=False):
    # Generates frequency distributions for all words in all corpora. Includes number of times a word was spoken and the number of speakers who spoke the word.
    dfs = []
    fdist_dict = {}
    for i in grades:
        print('Generating distributions for {}'.format(i))
        corpus = corpora[i]
        corpus_combined = ' '.join(corpus)

        fdist = dict(nltk.FreqDist(nltk.word_tokenize(corpus_combined)))
        fdist = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True))

        words = list(fdist.keys())
        n = len(words)
        for j in range(n):
            word = words[j]
            speakers = 0
            for k in corpus:
                if word in nltk.word_tokenize(k):
                    speakers += 1
            fdist[word] = {'freq':fdist[word], 'speakers':speakers}
            p = round(((j + 1) / n) * 100)
            print(f"'{word}' done: {p}% complete")

        fdist_df = []
        for j in fdist.items():
            fdist_df.append({'word':j[0],'freq':j[1]['freq'],'speakers':j[1]['speakers']})
        fdist_df = pd.DataFrame(fdist_df)
        fdist_df.columns = [j + '_' + i for j in fdist_df.columns]
        dfs.append(fdist_df)
        fdist_dict[i] = fdist

    if save:
        dfs = pd.concat(dfs, axis=1)
        dfs.to_excel('fdist_addon.xlsx')

        with open('fdist_addon.json', 'w') as file:
            json.dump(fdist_dict, file)
    
    return fdist_dict

def gen_sum_stats(type, save=False):
    # Generates summary statistics regarding the frequency distributions including the number of tokens, types, types greater than 1, types greater than 2, speakers greater than 2, and total number of speakers per grade. 
    corp_name = type + '_corpora'
    fdist_name = 'fdist_' + type

    adult_words = load_data('corpora_adult')
    adult_words = adult_words['unique_words']
    corpora = load_data(corp_name)
    remove_words = load_data('res')
    fdists = load_data(fdist_name)

    grades = [i for i in fdists]
    n_speakers = [len(corpora[i]) for i in grades]
    tokens = [sum([fdists[i][j]['freq'] for j in fdists[i]]) for i in fdists]
    types = [len(fdists[i]) for i in fdists]

    type_greater1 = []
    type_greater2 = []
    speakers_greater1 = []
    speakers_greater2 = []
    for i in fdists:
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        for j in fdists[i]:
            if j in adult_words and j not in remove_words:
                if fdists[i][j]['freq'] > 1:
                    x1 += 1
                if fdists[i][j]['freq'] > 2:
                    x2 += 1
                if fdists[i][j]['speakers'] > 1:
                    x3 += 1
                if fdists[i][j]['speakers'] > 2:
                    x4 += 1
        type_greater1.append(x1)
        type_greater2.append(x2)
        speakers_greater1.append(x3)
        speakers_greater2.append(x4)

    df = pd.concat([pd.DataFrame(i) for i in [grades, tokens, types, type_greater1, speakers_greater1, type_greater2, speakers_greater2, n_speakers]], axis=1).T
    cols = df.iloc[0]
    df = df[1:]
    df.columns = cols
    df.index = ['tokens','types','types > 1', 'speakers > 1', 'types > 2', 'speakers > 2', 'n_speakers']

    if save:
        df.to_excel('data/fdist_sum_' + type + '.xlsx')
    
    return df

def gen_word_groups(text, window):
    # Function for grouping words that surround another in a corpus
    word_groups = []
    words = []
    for i, word in enumerate(text):
        for w in range(window):
            # Getting the context that is ahead by *window* words
            if i + 1 + w < len(text): 
                word_groups.append([word] + [text[(i + 1 + w)]])
            # Getting the context that is behind by *window* words    
            if i - w - 1 >= 0:
                word_groups.append([word] + [text[(i - w - 1)]])
        if word not in words:
            words.append(word)
    result = {'word_groups':word_groups, 'words':words}
    return result
