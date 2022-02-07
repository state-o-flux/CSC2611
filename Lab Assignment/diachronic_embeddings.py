import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

#Step 1 Functions
data = pickle.load(open('embeddings/data.pkl', 'rb'))

def gen_data_dict(data):
    data_dict = {}
    for i in range(len(data['E'])):
        word = data['w'][i]
        data_dict[word] = {}
        for j in range(len(data['E'][i])):
            decade = data['d'][j]
            data_dict[word][decade] = data['E'][i][j]

    return data_dict

# Step 2 Functions 
def cosine_similarity(vec1, vec2):
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def mean_distance(data_dict):
    distances = {}
    for w in data_dict:
        w_dat = data_dict[w]
        decades = list(w_dat.keys())
        cos_dist = []
        for i in range(len(w_dat)-1):
            cos_dist.append(cosine_similarity(w_dat[decades[i]], w_dat[decades[i+1]]))
        distances[w] = np.mean(cos_dist)

    top_20 = dict(sorted(distances.items(), key=lambda item: item[1]))
    top_20 = {i[0]: i[1] for i in list(top_20.items())[:20]}

    bottom_20 = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
    bottom_20 = {i[0]: i[1] for i in list(bottom_20.items())[:20]}

    return {'distances':distances, 'most_change': top_20, 'least_change': bottom_20}

def max_distance(data_dict):
    distances = {}
    for w in data_dict:
        w_dat = data_dict[w]
        decades = list(w_dat.keys())
        cos_dist = []
        for i in range(len(w_dat)-1):
            cos_dist.append(cosine_similarity(w_dat[decades[i]], w_dat[decades[i+1]]))
        distances[w] = np.max(cos_dist)

    top_20 = dict(sorted(distances.items(), key=lambda item: item[1]))
    top_20 = {i[0]: i[1] for i in list(top_20.items())[:20]}

    bottom_20 = dict(sorted(distances.items(), key=lambda item: item[1], reverse=True))
    bottom_20 = {i[0]: i[1] for i in list(bottom_20.items())[:20]}

    return {'distances':distances, 'most_change': top_20, 'least_change': bottom_20}

def gen_decades_dict(data_dict):
    decades_dict = {}
    for w in data_dict:
        for dec in data_dict[w]:
            if dec not in decades_dict.keys():
                decades_dict[dec] = data_dict[w][dec]
            else:
                decades_dict[dec] = np.row_stack([decades_dict[dec],data_dict[w][dec]])
    return decades_dict

def align_decades(data_dict):
    decades_dict = gen_decades_dict(data_dict)

    base_decades = np.array([decades_dict[i] for i in decades_dict])
    align_decade = decades_dict[1990]

    aligned_decades = align_decade.dot(np.linalg.pinv(align_decade)).dot(base_decades)
    aligned_decades = np.array([aligned_decades[:,i,:] for i in range(aligned_decades.shape[1])])

    aligned_data_dict = {}
    for i in range(len(aligned_decades)):
        decade_array = aligned_decades[i]
        dec = data['d'][i]
        for j in range(len(decade_array)):
            word_vec = decade_array[j]
            word = data['w'][j]
            if word not in aligned_data_dict.keys():
                aligned_data_dict[word] = {dec: word_vec}
            else:
                aligned_data_dict[word][dec] = word_vec
    
    return aligned_data_dict

def run_semantic_change(data):
    data_dict = gen_data_dict(data)

    #Method 1: Raw mean cosine distance across decades
    raw_mean_distance = mean_distance(data_dict)
    #Method 2: Raw maximum distance across decades
    raw_max_distance = max_distance(data_dict)

    # Aligning the decades via linear alignment
    aligned_data_dict = align_decades(data_dict)
    #Method 3: Aligned mean cosine distance across decades
    aligned_mean_distance = mean_distance(aligned_data_dict)
    #Method 4: Aligned maximum distance across decades
    aligned_max_distance = max_distance(aligned_data_dict)


    most_change = {
        'method1_rawMean': raw_mean_distance['most_change'].keys(),
        'method2_rawMax': raw_max_distance['most_change'].keys(),
        'method3_alignMean': aligned_mean_distance['most_change'].keys(),
        'method4_alignMax': aligned_max_distance['most_change'].keys(),
        }
    least_change = {
        'method1_rawMean': raw_mean_distance['least_change'].keys(),
        'method2_rawMax': raw_max_distance['least_change'].keys(),
        'method3_alignMean': aligned_mean_distance['least_change'].keys(),
        'method4_alignMax': aligned_max_distance['least_change'].keys(),
        }
    change_values = {
        'method1_rawMean': raw_mean_distance['distances'].values(),
        'method2_rawMax': raw_max_distance['distances'].values(),
        'method3_alignMean': aligned_mean_distance['distances'].values(),
        'method4_alignMax': aligned_max_distance['distances'].values(),
        }

    most_change = pd.DataFrame(most_change)
    least_change = pd.DataFrame(least_change)
    change_values = pd.DataFrame(change_values)
    inter_correlations = round(change_values.corr(), 3)

    return {'most_change': most_change, 'least_change': least_change, 'inter_correlations': inter_correlations}

# Step 3 Functions
def score_change(distance_dict, change_words):

    distance_dict = dict(sorted(distance_dict['distances'].items(), key=lambda item: item[1]))

    distance_words = list(distance_dict.keys())
    word_index = []
    for i in range(len(distance_words)):
        if distance_words[i] in change_words:
            word_index.append(i)

    change_score = np.mean([(1-i/len(distance_words)) for i in word_index])
    return round(change_score, 4)

def evaluate_change(data):

    with open('change_words.txt', 'r') as file:
        change_words = file.readlines()

    change_words = [i.replace('\n','') for i in change_words]
    change_words = [i for i in change_words if i in data['w']]

    data_dict = gen_data_dict(data)
    aligned_data_dict = align_decades(data_dict)

    raw_mean = mean_distance(data_dict)
    raw_max = max_distance(data_dict)
    aligned_mean = mean_distance(aligned_data_dict)
    aligned_max = max_distance(aligned_data_dict)

    raw_mean = score_change(raw_mean, change_words)
    raw_max = score_change(raw_max, change_words)
    aligned_mean = score_change(aligned_mean, change_words)
    aligned_max = score_change(aligned_max, change_words)

    return {'raw_mean': raw_mean, 'raw_max': raw_max, 'aligned_mean': aligned_mean, 'aligned_max': aligned_max}

# Step 4 Functions
def get_distances(data_dict, word):
    distances = []
    decades = []
    for i in range(len(data_dict[word])-1):
        dec1 = data['d'][i]
        dec2 = data['d'][i+1]
        decades.append(round(np.mean([dec1, dec2])))
        distances.append(1-cosine_similarity(data_dict[word][dec1], data_dict[word][dec2]))

    for i in range(len(distances)):
        if distances[i] == np.max(distances):
            change_point = decades[i]
    
    return distances, decades, change_point

def plot_distances(data_dict, word):
    print("Timecourse of {}:".format(word))

    dist, dec, change_point = get_distances(data_dict, word)

    plt.subplots(1)
    plt.show
    plt.ylim(0.2, 0.8)
    plt.axvline(x=change_point, color='r')
    plt.text(change_point+1, 0.77, 'Change Point')
    plt.text(1965, 0.77, 'Word: {}'.format(word))
    plt.plot(dec,dist)
    return


### Results ###

# Step 2
semantic_change = run_semantic_change(data)
print("<--- Detecting Semantic Change --->")
print("Most Change:\n{}".format(semantic_change['most_change']))
print("Least Change:\n{}".format(semantic_change['least_change']))
print("Intercorrelations:\n{}".format(semantic_change['inter_correlations']))

# Step 3
change_scores = evaluate_change(data)
print("\n<--- Evaluting Accuracy of Semantic Change Detection --->")
print("Method 1 (raw mean): {}".format(change_scores['raw_mean']))
print("Method 2 (raw max): {}".format(change_scores['raw_max']))
print("Method 3 (aligned mean): {}".format(change_scores['aligned_mean']))
print("Method 4 (aligned max): {}".format(change_scores['aligned_max']))

# Step 4
data_dict = gen_data_dict(data)
print('\n<--- Visualization and Change Point Detection --->')
plot_distances(data_dict, 'harper')
plot_distances(data_dict, 'programs')
plot_distances(data_dict, 'peter')
