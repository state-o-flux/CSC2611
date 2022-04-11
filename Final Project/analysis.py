import json
import nltk
import numpy as np
import pandas as pd
from functools import reduce

from util import load_data, p_print, cosine_similarity, gen_z_score, gen_word_groups, unique

included_grades = ['00','01','02','03','04','05','06','full']

def gen_fdist(corpora, grades, save=False):
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

def gen_vocabulary(adult_corpus, by, minimum, save=False):
    # Generates the estimated vocabulary at each grade level.
    # Relies on the frequency distributions (fdists) computed from the time sliced corpora. The frequency distributions indicate the number of occurrences of each word at each time slice, as well as the the number of speakers who said each word at each time slice.
    # The vocabulary is determined by the presence of a word that satisfies the specified criteria at each time slice. The criteria is indicated by the 'by' and 'min' variables. by indicates which criteria to use, either 'freq' (the frequency) or 'speakers' (the number of speakers), and then min indicates the value equal to or above which the word is added to the vocabulary. For instance, if by = 'speakers' and min = 3, a word that has been spoken by 2 speakers at grade 1 will not be added to the learned vocabulary at grade 1, but if it is spoken by 3 speakers by grade 2, then it will be added to the learned vocabulary at grade 2.

    adult_words = adult_corpus['unique_words']
    fdists = load_data('fdist')
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

        writer = pd.ExcelWriter('vocabulary.xlsx', engine='xlsxwriter')
        vocabulary_df.to_excel(writer, sheet_name="Vocabulary")
        vocab_sum.to_excel(writer, sheet_name="Summary")
        writer.save()
    
    return vocabulary

def gen_context_model(corpus, grade, window):
    print(f'Generating Context Model for Grade: {grade} Window Size: {window}')
    corpus = corpus[grade]
    corpus = ' '.join(corpus)

    remove_words = load_data('res')

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
        p_print(i, n)
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
    return result

def export_context_models(corpora, window):
    context_model = {}
    for i in corpora:
        context_model[i] = gen_context_model(corpora, i, window)
    
    with open(f'data/context_model_{window}.json', 'w') as file:
        json.dump(context_model, file)

def gen_ppmi(context_models):
    result = {}
    if type(context_models) == list:
        for grade in context_models:
            print(f'Generating ppmi model from grade {grade} context model')
            context_model = context_models[grade]
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

            ppmi = {}
            n = len(words)
            for i in range(n):
                word = words[i]
                p_print(i, n)
                ppmi[word] = list(_pmi[i])
            print('\n')
            result[grade] = ppmi
    else:
        print(f'Generating ppmi model from context model')
        context_model = context_models
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
            p_print(i, n)
            result[word] = list(_pmi[i])
        print('\n')

    return result

def gen_word_degrees(window, gen_words, compare_words, ppmi, save=False):
    result = {}
    n = len(gen_words)
    print(f'Generating Degrees for {n} words in vocabulary')
    for i in range(n):
        word  = gen_words[i]
        degree = np.mean([cosine_similarity(ppmi[word], ppmi[j]) for j in compare_words if j != word])
        result[word] = degree
        p_print(i, n, word)

    if save:
        with open(f'data/word_degrees_{window}.json', 'w') as file:
            json.dump(result, file)
    
    return result

def gen_learnPotential(theory, grade, kids_corpora, adult_corpus, vocabulary, ppmi_model, degrees, z_threshold, save=False):
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

    kids_corpora = load_data('corpora_kids')
    adult_corpus = load_data('corpora_adult')

    context_models = load_data(f'context_model_{window}')

    if context_models is None:
        export_context_models(kids_corpora, window)
        context_models = load_data(f'context_model_{window}')

    ppmi_model = gen_ppmi(context_models['full'])
    vocabulary = gen_vocabulary(adult_corpus, by=vocab_by, minimum=vocab_minimum, save=True)

    degrees = load_data(f'word_degrees_{window}')
    if degrees is None:
        all_vocab_words = []
        for grade in vocabulary:
            all_vocab_words.extend([word for word in vocabulary[grade]])

        gen_word_degrees(window, all_vocab_words, all_vocab_words, ppmi_model, save=True)

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

    writer = pd.ExcelWriter(f'learningPotential_{window}.xlsx', engine='xlsxwriter')
    lp_scores.to_excel(writer, sheet_name="Scores", index=False)
    lp_summ.to_excel(writer, sheet_name="Summary", index=False)
    writer.save()

    return

main(window=4, vocab_by='speakers', vocab_minimum=3, z_threshold=1.5)