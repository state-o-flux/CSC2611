# Step 1
import nltk
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from nltk.corpus import brown

words = brown.words()
words = [i.lower() for i in words if i.isalpha()]

# Step 2
fdist = nltk.FreqDist(w for w in words if w.isalpha())
fdist = dict(fdist)

fdist = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True))
fdist['serf'] = 0

#top 5000
top5000 = {i[0]:i[1] for i in list(fdist.items())[:5000]}

top5 = {i[0]:i[1] for i in list(fdist.items())[:5]}
bottom5 = {i[0]:i[1] for i in list(fdist.items())[-5:]}

RG65 = ['cord', 'smile', 'rooster', 'voyage', 'noon', 'string', 'fruit', 'furnace', 'autograph', 'shore', 'woodland', 'automobile', 'wizard', 'mound', 'stove', 'grin', 'implement', 'asylum', 'monk', 'graveyard', 'glass', 'magician', 'boy', 'cushion', 'jewel', 'slave', 'cemetery', 'coast', 'forest', 'lad', 'oracle', 'sage', 'food', 'bird', 'hill', 'crane', 'car', 'journey', 'brother', 'cock', 'madhouse', 'tumbler', 'signature', 'tool', 'pillow', 'midday', 'gem']

for i in RG65:
    if i not in top5000.keys():
        top5000[i] = fdist[i]

W = len(top5000)

# Step 3
# Generates the word-context model, took over two hours to run so storing it as a document for import from now on
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

M1 = pd.read_csv('word_vector_model.csv', index_col=0)

# Step 4
# My version of calculating pmi
def pmi(df):
    arr = df.values

    shape = arr.shape
    W = arr.sum()
    row_sums = arr.sum(axis=1)

    ppmi_total = []
    for i in range(shape[0]):
        ppmi_i = []
        for j in range(shape[1]):
            if i == j:
                ppmi_i.append(0)
            ppmi = max([np.log2(((arr[i][j]) / W) / ((row_sums[i]/W) * (row_sums[j]/W))),0])
            ppmi_i.append(ppmi)
        ppmi_total.append(ppmi_i)
    
    result = np.array(ppmi_total)
    result = pd.DataFrame(result)
    return result
# PMI function found on the internet (much faster)
def pmi(df):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    arr = df.values

    # p(y|x) probability of each t1 overlap within the row
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T

    # p(y) probability of each t1 in the total set
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    # PMI: log( p(y|x) / p(y) )
    # This is the same data, normalized
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log2(ratio)
    _pmi[_pmi < 0] = 0

    return _pmi

M1_plus = pd.DataFrame(pmi(M1))
M1_plus.columns = top5000.keys()
M1_plus.index = top5000.keys()


#Step 5
# My own function
def pca_3ways(M1_plus):
    X_meaned = M1_plus - np.mean(M1_plus, axis = 0)
    cov_mat = np.cov(X_meaned, rowvar = False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    pca_10 = sorted_eigenvectors[:,0:10]
    pca_100 = sorted_eigenvectors[:,0:100]
    pca_300 = sorted_eigenvectors[:,0:300]

    M2_10 = np.dot(pca_10.transpose(), X_meaned.transpose()).transpose()
    M2_100 = np.dot(pca_100.transpose(), X_meaned.transpose()).transpose()
    M2_300 = np.dot(pca_300.transpose(), X_meaned.transpose()).transpose()

    return M2_10, M2_100, M2_300
#M2 = pca_3ways(M1_plus)

# Pre-made functions
pca_10 = PCA(n_components=10)
pca_10 = pca_10.fit(M1_plus)
M2_10 = pca_10.transform(M1_plus)

pca_100 = PCA(n_components=100)
pca_100 = pca_100.fit(M1_plus)
M2_100 = pca_100.transform(M1_plus)

pca_300 = PCA(n_components=300)
pca_300 = pca_300.fit(M1_plus)
M2_300 = pca_300.transform(M1_plus)

# Step 6
P = [
    ['cord','smile'],
    ['rooster','voyage'],
    ['noon','string'],
    ['fruit','furnace'],
    ['autograph','shore'],
    ['automobile','wizard'],
    ['mound','stove'],
    ['grin','implement'],
    ['asylum','fruit'],
    ['asylum', 'monk'],
    ['graveyard','madhouse'],
    ['glass','magician'],
    ['boy','rooster'],
    ['cushion','jewel'],
    ['monk','slave'],
    ['asylum','cemetery'],
    ['coast','forest'],
    ['grin','lad'],
    ['shore','woodland'],
    ['monk','oracle'],
    ['boy','sage'],
    ['automobile','cushion'],
    ['mound','shore'],
    ['lad','wizard'],
    ['forest','graveyard'],
    ['food','rooster'],
    ['cemetery','woodland'],
    ['shore','voyage'],
    ['bird','woodland'],
    ['coast','hill'],
    ['furnace','implement'],
    ['crane','rooster'],
    ['hill','woodland'],
    ['car','journey'],
    ['cemetery','mound'],
    ['glass','jewel'],
    ['magician','oracle'],
    ['crane','implement'],
    ['brother','lad'],
    ['sage','wizard'],
    ['oracle','sage'],
    ['bird','crane'],
    ['bird','cock'],
    ['food','fruit'],
    ['brother','monk'],
    ['asylum','madhouse'],
    ['furnace','stove'],
    ['magician','wizard'],
    ['hill','mound'],
    ['cord','string'],
    ['glass','tumbler'],
    ['grin','smile'],
    ['journey','voyage'],
    ['autograph','signature'],
    ['coast','shore'],
    ['forest','woodland'],
    ['implement','tool'],
    ['cock','rooster'],
    ['boy','lad'],
    ['cushion','pillow'],
    ['cemetery','graveyard'],
    ['automobile','car'],
    ['midday','noon'],
    ['gem','jewel']
]
S = [.02,.04,.04,.05,.06,.11,.14,.18,.19,.39,.42,.44,.44,.45,.57,.79,.85,.88,.9,.91,.96,.97,.97,.99,1,1.09,1.18,1.22,1.24,1.26,1.37,1.41,1.48,1.55,1.69,1.78,1.82,2.37,2.41,2.46,2.61,2.63,2.63,2.69,2.74,3.04,3.11,3.21,3.29,3.41,3.45,3.46,3.58,3.59,3.6,3.65,3.66,3.68,3.82,3.84,3.88,3.92,3.94,3.94]

# Step 7
def cosine_similarity(df, pair):
    v1 = np.array(df[pair[0]])
    v2 = np.array(df[pair[1]])

    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def create_importance_dataframe(pca, df):
    importance_df = pd.DataFrame(pca.components_)
    importance_df.columns = df.columns
    importance_df = importance_df.apply(np.abs)
    importance_df = importance_df.transpose()
    new_columns = [f'PC{i}' for i in range(1, importance_df.shape[1] + 1)]
    importance_df.columns = new_columns
    return importance_df

def which(vec):
    ind = []
    for i in range(len(vec)):
        if vec[i]:
            ind.append(i)
    return ind

def which_component(pca_importance, word):
    word_importance = list(pca_importance.loc[word,:])
    word_component = which([i == max(word_importance) for i in word_importance])[0]
    return word_component

def pca_cosine_similarity(orig_df, pca_df, pca_model, pairs):

    pca_importance = create_importance_dataframe(pca_model, orig_df)

    cos_sim = []
    for i in pairs:
        v1 = pca_df[:,which_component(pca_importance, i[0])]
        v2 = pca_df[:,which_component(pca_importance, i[1])]
        cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_sim.append(cs)
    
    return cos_sim

M1_cosSim = [cosine_similarity(M1, i) for i in P]
M1_plus_cosSim = [cosine_similarity(M1_plus, i) for i in P]
M2_10_cosSim = pca_cosine_similarity(M1_plus, M2_10, pca_10, P)
M2_100_cosSim = pca_cosine_similarity(M1_plus, M2_100, pca_100, P)
M2_300_cosSim = pca_cosine_similarity(M1_plus, M2_300, pca_300, P)

# Step 8
M1_corr = np.corrcoef(S, M1_cosSim)[0, 1]
M1_plus_corr = np.corrcoef(S, M1_plus_cosSim)[0, 1]
M2_10_corr = np.corrcoef(S, M2_10_cosSim)[0, 1]
M2_100_corr = np.corrcoef(S, M2_100_cosSim)[0, 1]
M2_300_corr = np.corrcoef(S, M2_300_cosSim)[0, 1]