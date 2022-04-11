# CSC2611 - Final Project Code Repository

Included here is all the code needed to run the models involved in the report titled: "Longitudinal Analysis of Developmental Semantic Networks: Hypothesis Testing Using a Distributional Model".

In order to replicate the results reported you can simply run the two analysis files (analysis.py and analysis.r). analysis.py takes over an hour to run so the resulting data has also been uploaded so as to save time (learningPotential_2_speakers_3.xlsx). With this data file, analysis.r can be run to see all the results in the report.

There are also four parameters that can be altered generating different results:
* window: The window size to be used when computing the context model
* vocab_by: Can be one of 'freq' or 'speakers', used to determine what the vocabulary is based on. If vocab_by == 'freq' the vocabulary is determined by the frequency with which the word occurred in the corpus. If vocab_by == 'speakers' the vocabulary is determined by the number of speakers who uttered the word in the corpus.
* vocab_minimum: The minimum value to be used to determine the vocabulary, either of frequency or speakers.
* z_threshold: The z value threshold to use for determining whether two words are connected under t1 (preferential attachment model). Must be between -3 and 3.

If you want to alter any of these parameters, just run the main function from analysis.py with your desired parameters, and then the test_theories function from analysis.r with the same parameters.
