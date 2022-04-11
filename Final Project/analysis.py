import json
import nltk
import numpy as np
import pandas as pd
from functools import reduce

from util import load_data, p_print, cosine_similarity, gen_z_score, gen_word_groups, unique, gen_fdist

included_grades = ['00','01','02','03','04','05','06','full']

def gen_vocabulary(adult_corpus, window, by, minimum, save=False):
    # Generates the estimated vocabulary at each grade level.
    # Relies on the frequency distributions (fdists) computed from the time sliced corpora. The frequency distributions indicate the number of occurrences of each word at each time slice, as well as the the number of speakers who said each word at each time slice.
    # The vocabulary is determined by the presence of a word that satisfies the specified criteria at each time slice. The criteria is indicated by the 'by' and 'min' variables. by indicates which criteria to use, either 'freq' (the frequency) or 'speakers' (the number of speakers), and then min indicates the value equal to or above which the word is added to the vocabulary. For instance, if by = 'speakers' and min = 3, a word that has been spoken by 2 speakers at grade 1 will not be added to the learned vocabulary at grade 1, but if it is spoken by 3 speakers by grade 2, then it will be added to the learned vocabulary at grade 2.

    adult_words = adult_corpus['unique_words']
    fdists = load_data('fdist')
    if fdists is None:
        kids_corpora = load_data('corpora_kids')
        for grade in kids_corpora:
            fdists = gen_fdist(kids_corpora, grade, save=True)

    remove_words = load_data('res')

    final_vocabulary = []
    vocabulary = {}
    for i in fdists:
        vocabulary[i] = []
        for j in fdists[i]:
            if fdists[i][j][by] >= minimum and j in adult_words and j not in remove_words and j not in final_vocabulary:
                final_vocabulary.append(j)
                vocabulary[i].append(j)
    
    vocab_sum = {}
    vocab_keys = list(vocabulary.keys())
    for i in range(len(vocabulary)):
        additional = len(vocabulary[vocab_keys[i]])
        if i == 0:
            base = additional
            additional = 0
        elif i == 1:
            base = vocab_sum[vocab_keys[i-1]]['base']
        else:
            base = vocab_sum[vocab_keys[i-1]]['base'] + vocab_sum[vocab_keys[i-1]]['additional']
        vocab_sum[vocab_keys[i]] = {'base':base, 'additional':additional}

    if save:
        vocabulary_df = pd.concat([pd.DataFrame(vocabulary[i]) for i in vocabulary], axis=1)
        vocabulary_df.columns = vocabulary.keys()
        vocab_sum = pd.DataFrame(vocab_sum)

        writer = pd.ExcelWriter(f'vocabulary_{window}_{by}_{minimum}.xlsx', engine='xlsxwriter')
        vocabulary_df.to_excel(writer, sheet_name="Vocabulary")
        vocab_sum.to_excel(writer, sheet_name="Summary")
        writer.save()
    
    return vocabulary

def gen_context_model(corpus, grade, window, save=False):
    # Generates the context model (also known as the word x word matrix) from the corpus at a specific grade. Corpora must be a grade tagged dictionary in which the keys are grades and each value is a corpus from that specific grade. Window indicates the window size to use for the context model.
    corpus = corpus[grade]
    corpus = ' '.join(corpus)
    remove_words = load_data('res')

    print(f'Generating Context Model Window Size: {window}')

    text = nltk.word_tokenize(corpus)
    text = [i for i in text if i not in remove_words]

    word_groups = gen_word_groups(text, window)

    words = word_groups['words']
    words.sort()
    word_groups = word_groups['word_groups']

    n = len(words)
    context_model = {}
    for i in range(n):
        word = words[i]
        match_words = {}
        for j in word_groups:
            if word in j:
                for k in j:
                    if k in match_words.keys():
                        match_words[k] += 1
                    else:
                        match_words[k] = 1
        p_print(i, n, word)
        context_model[word] = match_words

    result = {}
    for i in words:
        i_val = []
        for j in words:
            if i == j:
                i_val.append(0)
            elif j in context_model[i]:
                i_val.append(context_model[i][j])
            else:
                i_val.append(0)
        result[i] = i_val
    
    print('\n')

    if save:
        with open(f'data/context_model_{window}.json', 'w') as file:
            json.dump(result, file)

    return result

def gen_ppmi(context_model):
    # Generates the positive pointwise mututal information model from the context model.
    print(f'Generating ppmi model from context model')

    result = {}
    words = list(context_model.keys())
    df = pd.DataFrame(context_model)
    arr = df.values

    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T

    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log2(ratio)
    _pmi[_pmi < 0] = 0

    n = len(words)
    for i in range(n):
        word = words[i]
        p_print(i, n, word)
        result[word] = list(_pmi[i])
    print('\n')

    return result

def gen_word_degrees(window, vocab_by, vocab_minimum, gen_words, compare_words, ppmi, save=False):
    # Generates the word degree value (connectiveness) of each word as derived from the ppmi model. Degree is computed for every gen_word as the average cosine_similarity of that word with each compare_word.
    result = {}
    n = len(gen_words)
    print(f'Generating Degrees for {n} words in vocabulary')
    for i in range(n):
        word  = gen_words[i]
        degree = np.mean([cosine_similarity(ppmi[word], ppmi[j]) for j in compare_words if j != word])
        result[word] = degree
        p_print(i, n, word)

    if save:
        file_name = f'data/word_degrees_{window}_{vocab_by}_{vocab_minimum}.json'
        with open(file_name, 'w') as file:
            json.dump(result, file)
    
    return result

def gen_learnPotential(theory, grade, kids_corpora, adult_corpus, vocabulary, ppmi_model, degrees, z_threshold, save=False):
    # Generates the learning potential score of each word based on a given theory. Theories are described below.
    # t1: Theory 1: PREFERENTIAL ATTACHMENT
        # Words learned at t + 1 should be those that connect to the higher-degree known words.
        # LP score = mean degree of the known words that the new word attaches to.
    # t2: Theory 2: PREFERENTIAL ACQUISITION
        # Words learned should be the highest degree words left to learn.
        # LP score = degree of the word in the presumed learning environment.
    # t3: Theory 3: LURE OF ASSOCIATES
        # Words learned at t + 1 should be those that most connect to known words.
        # LP score = mean cosine similarity of the new word to all known words.
    # Baseline: WORD FREQUENCY 
        # Words learned should be the words with the greatest frequency.
        # LP score = frequency of occurence of the word in the presumed learning environment.
    theories = {'t1': "Preferential Attachment", "t2": "Preferential Acquisition", "t3": "Lure of Associates", "baseline": "Baseline Frequency"}
    print(f'Generating {theories[theory]} Learning Potential scores for Grade {grade}')

    full_corpus_kids = kids_corpora['full']
    full_corpus_kids = ' '.join(full_corpus_kids)
    full_corpus_adults = adult_corpus['corpus']

    degree_vals = list(degrees.values())
    degree_z_scores = gen_z_score(degree_vals)
    degree_z_scores = {list(degrees.keys())[i]: degree_z_scores[i] for i in range(len(degrees))}
    raw_threshold = np.mean(degree_vals) + (z_threshold * np.std(degree_vals))

    earlier_grades = []
    for i in vocabulary.keys():
        if i == grade:
            break
        earlier_grades.append(i)

    later_grades = [i for i in vocabulary.keys() if i not in earlier_grades]

    known_words = [vocabulary[i] for i in earlier_grades]
    known_words = sum(known_words, [])

    unknown_words = [vocabulary[i] for i in later_grades]
    unknown_words = sum(unknown_words, [])

    learned_words = [int(i in vocabulary[grade]) for i in unknown_words]

    all_words_adults = nltk.word_tokenize(full_corpus_adults)

    learnPotential = []
    n = len(unknown_words)
    for i in range(n):
        word = unknown_words[i]
        p_print(i, n, word)
        if theory == "t1":
            connected_words = [j for j in known_words if j in ppmi_model.keys() and cosine_similarity(ppmi_model[word], ppmi_model[j]) > raw_threshold]
            learnP = np.mean([degrees[j] for j in connected_words])
            learnPotential.append(learnP)
        elif theory == "t2":
            learnPotential.append(degrees[word])
        elif theory == "t3":
            learnP = np.mean([cosine_similarity(ppmi_model[word], ppmi_model[j]) for j in known_words])
            learnPotential.append(learnP)
        elif theory == "baseline":
            learnPotential.append(all_words_adults.count(word))
    
    learnPotential_z_scores = gen_z_score(learnPotential)
    
    result = pd.DataFrame(list(zip([grade] * len(unknown_words), unknown_words, learned_words, learnPotential_z_scores)), columns = ['grade', 'unknownWord', 'learned', 'learnPotential'])

    print(f'\n{theories[theory]} Learning Potential scores generated for Grade {grade}')

    if save:
        name = f'learnPotential_{grade}.xlsx'
        result.to_excel(name)

    return result

def summarize_learnPotential(df, lp_score):
    # Summarizes the learn potential scores by computing the average score per word and the grade at which the word was learned.
    words = unique(list(df['unknownWord']))

    summ_df = []
    for word in words:
        word_df = df[df['unknownWord'] == word]
        mean_lp = np.mean(list(word_df[lp_score]))
        learned_df = word_df[word_df['learned'] == 1]
        if len(learned_df) == 0:
            learned = "Never Learned"
        else:
            learned = list(learned_df.grade)[0]
        summ_df.append([word, learned, mean_lp])

    summ_df = pd.DataFrame(summ_df)
    summ_df.columns = ['word','learned',lp_score]
    return summ_df

def main(window, vocab_by, vocab_minimum, z_threshold):
    # Runs all the results and generates two outputs, a vocabulary file and a learnPotential file each include the window size, vocab_by and vocab_minimum values in te file name. The vocabulary file contains the estimated vocabulary which includes all of the words which are determined to be known at each grade level. The learningPotential file includes a learning potential score from each theory for each word at each grade and whether the word was learned or not.
    # Parameters determine the following:
        # window: The window size to be used when computing the context model
        # vocab_by: Can be one of 'freq' or 'speakers', used to determine what the vocabulary is based on. If vocab_by == 'freq' the vocabulary is determined by the frequency with which the word occurred in the corpus. If vocab_by == 'speakers' the vocabulary is determined by the number of speakers who uttered the word in the corpus.
        # vocab_minimum: The minimum value to be used to determine the vocabulary, either of frequency or speakers.
        # z_threshold: The z value threshold to use for determining whether two words are connected under t1 (preferential attachment model). Must be between -3 and 3.

    kids_corpora = load_data('corpora_kids')
    adult_corpus = load_data('corpora_adult')

    context_model = load_data(f'context_model_{window}')
    if context_model is None:
        gen_context_model(kids_corpora, 'full', window, save=True)
        context_model = load_data(f'context_model_{window}')

    ppmi_model = gen_ppmi(context_model)
    vocabulary = gen_vocabulary(adult_corpus, window, vocab_by, vocab_minimum, save=True)

    degrees = load_data(f'word_degrees_{window}_{vocab_by}_{vocab_minimum}')
    if degrees is None:
        all_vocab_words = []
        for grade in vocabulary:
            all_vocab_words.extend([word for word in vocabulary[grade]])

        gen_word_degrees(window, vocab_by, vocab_minimum, all_vocab_words, all_vocab_words, ppmi_model, save=True)
        degrees = load_data(f'word_degrees_{window}_{vocab_by}_{vocab_minimum}')

    lp_scores = {}
    lp_grades = ['01','02','03','04','05','06']
    test_theories = ['t1','t2', 't3', 'baseline']
    for t in test_theories:
        t_scores = []
        for grade in lp_grades:
            lp_score = gen_learnPotential(t, grade, kids_corpora, adult_corpus, vocabulary, ppmi_model, degrees, z_threshold)
            t_scores.append(lp_score)
        lp_scores[t] = t_scores

    lp_dfs = [pd.concat(lp_scores[i]) for i in lp_scores]
    lp_scores = reduce(lambda df1,df2: pd.merge(df1,df2,on=['grade','unknownWord','learned']), lp_dfs)

    col_names = ['t1_LP','t2_LP','t3_LP','baseline_LP']
    lp_cols = list(lp_scores.columns[:-4])
    lp_cols.extend(col_names)
    lp_scores.columns = lp_cols

    lp_summ = []
    for i in col_names:
        lp_summ.append(summarize_learnPotential(lp_scores, i))

    lp_summ = reduce(lambda df1,df2: pd.merge(df1,df2,on=['word','learned']), lp_summ)

    writer = pd.ExcelWriter(f'learningPotential_{window}_{vocab_by}_{vocab_minimum}.xlsx', engine='xlsxwriter')
    lp_scores.to_excel(writer, sheet_name="Scores", index=False)
    lp_summ.to_excel(writer, sheet_name="Summary", index=False)
    writer.save()

    return

if __name__ == '__main__':
    main(window=2, vocab_by='speakers', vocab_minimum=3, z_threshold=1.5)
