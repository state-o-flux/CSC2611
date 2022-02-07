import nltk
import json
import numpy as np
import pandas as pd
from nltk.corpus import brown
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# Step 2 Functions
def get_corp():
    words = brown.words()
    words = [i.lower() for i in words if i.isalpha()]

    # Step 2
    fdist = nltk.FreqDist(w for w in words if w.isalpha())
    fdist = dict(fdist)

    fdist = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True))
    fdist['serf'] = 0

    #top 5000
    top5000 = {i[0]:i[1] for i in list(fdist.items())[:5000]}

    synonyms = json.load(open('synonyms.json','r'))
    for i in synonyms:
        for j in synonyms['words']:
            if j not in top5000.keys():
                top5000[j] = fdist[j]

    return {'words':words, 'top5000':top5000}

def gen_word_context_model(words, top5000):
    all_matches = []
    for i in top5000:
        i_matches = []
        i_ind = np.where(np.array(words) == i)[0]
        x = 0
        for j in top5000.keys():
            if j == i:
                i_matches.append(0)
                continue
            ij_match = 0
            for k in i_ind:
                if words[k-1] == j:
                    ij_match +=1
                if words[k+1] == j:
                    ij_match += 1
            i_matches.append(ij_match)
            if x % 1000 == 0:
                print('{} {}: {}'.format(x, j, ij_match))
            x+=1
        print('---> {} Complete'.format(i))
        all_matches.append(i_matches)
    return all_matches

def pmi(df):
    arr = df.values

    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T

    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log2(ratio)
    _pmi[_pmi < 0] = 0

    return _pmi

def pca_projection(df, n_components):
    pca_model = PCA(n_components=n_components)
    pca_model.fit(df)
    df_transformed = pca_model.transform(df)
    df_projected = pca_model.inverse_transform(df_transformed)
    df_projected = pd.DataFrame(df_projected)
    df_projected.columns = df.columns
    return df_projected

def get_model(type):
    if type == 'word2vec':
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
    elif type == 'word_context':
        words = get_corp()
        # Generates the word-context model, takes over two hours to run so stored it as a csv for import after running
        #model = gen_word_context_model(words['words'], top5000['top5000])
        # Split csv in half in order to upload to github
        model_1 = pd.read_csv('word_vector_model_1.csv', index_col=0)
        model_2 = pd.read_csv('word_vector_model_2.csv', index_col=0)
        model = pd.concat([model_1, model_2])
    elif type == 'pmi_word_context':
        words = get_corp()
        model_1 = pd.read_csv('word_vector_model_1.csv', index_col=0)
        model_2 = pd.read_csv('word_vector_model_2.csv', index_col=0)
        model = pd.concat([model_1, model_2])
        model = pd.DataFrame(pmi(model))
        model.columns = words['top5000'].keys()
        model.index = words['top5000'].keys()
    elif type == 'lsa':
        words = get_corp()
        model_1 = pd.read_csv('word_vector_model_1.csv', index_col=0)
        model_2 = pd.read_csv('word_vector_model_2.csv', index_col=0)
        model = pd.concat([model_1, model_2])
        model = pd.DataFrame(pmi(model))
        model.columns = words['top5000'].keys()
        model.index = words['top5000'].keys()
        model = pca_projection(model, n_components=100)
    return model

# Step 3 Functions
def cosine_similarity(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def test_similarity(model):
    synonyms = json.load(open('synonyms.json','r'))
    pairs = [i['pair'] for i in synonyms['pairs'].values()]

    human_synonym_scores = [i['value'] for i in synonyms['pairs'].values()]
    word2vec_synonym_scores = [cosine_similarity(model[i[0]], model[i[1]]) for i in pairs]

    human_word2vec_corr = np.corrcoef(human_synonym_scores, word2vec_synonym_scores)[0, 1]

    return human_word2vec_corr

# Step 4 Functions
def get_top_analogy_w2v(model, comparative_word, isto_word, sub_word):
    result = model.most_similar(negative=[comparative_word],
                                positive=[isto_word, sub_word])
    return result[0][0]

def word_context_model(words, top5000):
    all_matches = []
    for i in top5000:
        i_matches = []
        i_ind = np.where(np.array(words) == i)[0]
        x = 0
        for j in top5000.keys():
            if j == i:
                i_matches.append(0)
                continue
            ij_match = 0
            for k in i_ind:
                if words[k-1] == j:
                    ij_match +=1
                if words[k+1] == j:
                    ij_match += 1
            i_matches.append(ij_match)
            if x % 1000 == 0:
                print('{} {}: {}'.format(x, j, ij_match))
            x+=1
        print('---> {} Complete'.format(i))
        all_matches.append(i_matches)
    return all_matches

def lsa_300():
    words = brown.words()
    words = [i.lower() for i in words if i.isalpha()]

    fdist = nltk.FreqDist(w for w in words if w.isalpha())
    fdist = dict(fdist)

    # word_context_model takes over 2 hours to run so I exported and stored it as a csv to save time (had to store it as two different csv files to upload to github)
    #top5000 = {i[0]:i[1] for i in list(fdist.items())[:5000]}
    #M1 = word_context_model(words, top5000)

    M1_1 = pd.read_csv('word_vector_model_1.csv', index_col=0)
    M1_2 = pd.read_csv('word_vector_model_2.csv', index_col=0)
    M1 = pd.concat([M1_1, M1_2])

    M1_plus = pd.DataFrame(pmi(M1))
    M1_plus.columns = M1.columns
    M1_300 = pca_projection(M1_plus, 300)

    return M1_300

def most_similar(df, vec):
    similarity_scores = {}
    for i in df.columns:
        similarity = cosine_similarity(df[i], vec)
        similarity_scores[i] = similarity
    
    similarity_scores = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True))

    top_10 = {i[0]: i[1] for i in list(similarity_scores.items())[:10]}
    return top_10

def get_top_analogy_lsa(df, comparative_word, isto_word, sub_word):
    target_vector = np.array(df[isto_word]) - np.array(df[comparative_word]) + np.array(df[sub_word])
    top_10 = most_similar(df, target_vector)

    for i in top_10:
        if i not in [comparative_word, isto_word, sub_word]:
            break

    return i

def gen_test_set(analogies, LSA_300):
    in_LSA = []
    for index, row in analogies.iterrows():
        words = [row['comparative_word'], row['is_to'], row['sub_word'], row['target_word']]
        in_LSA.append(all([i in LSA_300 for i in words]))

    in_LSA = pd.DataFrame(in_LSA)
    in_LSA.columns = ['in_LSA']

    test_set = pd.concat([analogies, in_LSA], axis = 1)
    test_set = test_set[test_set['in_LSA'] == True]

    return test_set

def test_analogy(model, analogies, sub_col, sub_val, type):

    subset = analogies[analogies[sub_col] == sub_val]
    subset = subset[['comparative_word','is_to','sub_word','target_word']]
    subset = subset.values.tolist()

    result = []
    for i in subset:
        if type == 'w2v':
            prediction = get_top_analogy_w2v(model, i[0], i[1], i[2])
        elif type == 'lsa':
            prediction = get_top_analogy_lsa(model, i[0], i[1], i[2])
        correct = prediction == i[3]
        result.append([i[0], i[1], i[2], i[3], prediction, correct])

    result = pd.DataFrame(result)
    result.columns = ['comparative_word', 'is_to', 'sub_word', 'target_word', 'prediction', 'correct']
    
    num_correct = sum(result['correct'])
    out_of = result.shape[0]
    accuracy = round(num_correct / out_of, 4)
    
    return {'df':result, 'num_correct':num_correct, 'out_of':out_of, 'accuracy':accuracy}


### Results ###

# Step 3
print("<--- Synonym Detection Task --->")
print("Human - Word2Vec correlation: {}".format(test_similarity(get_model('word2vec'))))
print("Human - Word Context Vector correlation: {}".format(test_similarity(get_model('word_context'))))
print("Human - PMI Word Context Vector correlation: {}".format(test_similarity(get_model('pmi_word_context'))))
print("Human - LSA with 100 Principal Components (best LSA model) correlation: {}".format(test_similarity(get_model('lsa'))))

# Step 4
print("\n<--- Analogic Reasoning Task --->")
w2v_model = get_model('word2vec')
analogies = pd.read_csv('analogies.csv')
LSA_300 = lsa_300()

test_set = gen_test_set(analogies, LSA_300)

w2v_semantic = test_analogy(w2v_model, test_set, 'category', 'semantic', 'w2v')
print("Word2Vec semantic: {} correct out of {} = {}% accuracy".format(w2v_semantic['num_correct'], w2v_semantic['out_of'], round(w2v_semantic['accuracy']*100, 2)))

w2v_syntactic = test_analogy(w2v_model, test_set, 'category', 'syntactic', 'w2v')
print("Word2Vec semantic: {} correct out of {} = {}% accuracy".format(w2v_syntactic['num_correct'], w2v_syntactic['out_of'], round(w2v_syntactic['accuracy']*100, 2)))

lsa_semantic = test_analogy(LSA_300, test_set, 'category', 'semantic', 'lsa')
print("LSA semantic: {} correct out of {} = {}% accuracy".format(lsa_semantic['num_correct'], lsa_semantic['out_of'], round(lsa_semantic['accuracy']*100, 2)))

lsa_syntactic = test_analogy(LSA_300, test_set, 'category', 'syntactic', 'lsa') # 0.1151
print("Word2Vec semantic: {} correct out of {} = {}% accuracy".format(lsa_syntactic['num_correct'], lsa_syntactic['out_of'], round(lsa_syntactic['accuracy']*100, 2)))
