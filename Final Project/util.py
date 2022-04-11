import sys
import json
import numpy as np

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
            None
    elif 'word_degrees' in file_name:
        try:
            with open(f'data/{file_name}.json', 'r') as file:
                data = json.load(file)
        except:
            None
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
    z_score = [(i - np.mean(vec)) / np.std(vec) for i in vec]
    return z_score

def gen_word_groups(text, window):
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
