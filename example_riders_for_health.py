#!/usr/bin/env python
# encoding: utf-8
"""
example.py

Created by Ben Birnbaum on 2012-12-02.
benjamin.birnbaum@gmail.com

Example use of outlierdetect.py module.
"""

from __future__ import print_function
from matplotlib import mlab
import outlierdetect
import pandas as pd
import sys
import numpy as np
from collections import OrderedDict

DATA_FILE = 'Sample_Transport_Log_2022-10-21.xlsx-Forms.csv'
INTERVIEWER_ID = 'Courier ID'

def compute_mma(data, QUESTIONS, action):
    # Compute MMA outlier scores.
    (mma_scores, _) = outlierdetect.run_mma(data, INTERVIEWER_ID, QUESTIONS, ['---'])
    print("\nMMA outlier scores")
    results = print_scores(mma_scores, action)
    return results

def compute_sva(data, QUESTIONS, action):
    # Compute SVA outlier scores.
    (sva_scores, _) = outlierdetect.run_sva(data, INTERVIEWER_ID, QUESTIONS, ['---'])
    print("\nSVA outlier scores")
    results = print_scores(sva_scores, action)
    return results

def _normalize_counts(counts, val=1):
    """Normalizes a dictionary of counts, such as those returned by _get_frequencies().

    Args:
        counts: a dictionary mapping value -> count.
        val: the number the counts should add up to.

    Returns:
        dictionary of the same form as counts, except where the counts have been normalized to sum
        to val.
    """
    n = sum(counts.values())
    if n == 0:
        return counts
    frequencies = {}
    for r in list(counts.keys()):
        frequencies[r] = val * float(counts[r]) / float(n)
    return frequencies

def print_scores(scores, action):
    results = []
    for interviewer in scores.keys():
        print('\n')
        print("============================== %s ==================================" % interviewer)
        for column in scores[interviewer].keys():
            
            score = scores[interviewer][column]['score']

            print("Question: %s" % str(column))
            print("Score: %s" % str(score))

            # Uncomment the following to print additional information about each outlier:
            observed_frequencies = scores[interviewer][column]['observed_freq']
            expected_frequencies = scores[interviewer][column]['expected_freq']
            p_value = scores[interviewer][column]['p_value']
            # The outlier algorithm doesn't return these sorted or normalized making debugging a bit tricky
            if "".join(list(observed_frequencies.keys())).isdigit():
                observed_frequencies = {int(k): v for k, v in observed_frequencies.items()}
            if "".join(list(expected_frequencies.keys())).isdigit():
                expected_frequencies = {int(k): v for k, v in expected_frequencies.items()}
            observed_frequencies = dict(OrderedDict(sorted(observed_frequencies.items())))
            expected_frequencies = dict(OrderedDict(sorted(expected_frequencies.items())))

            observed_frequencies_norm = _normalize_counts(observed_frequencies , val=1)
            expected_frequencies_norm = _normalize_counts(expected_frequencies , val=1)

            print("Observed Frequencies: %s" % observed_frequencies)
            print("Expected Frequencies: %s" % expected_frequencies)
            print("Observed Frequencies normalized: %s" % observed_frequencies_norm)
            print("Expected Frequencies normalized: %s" % expected_frequencies_norm)
            print("P-Value: %d" % p_value)
            print('\n')

            results.append({
                "interviewer":interviewer,
                "action": action,
                "question": str(column),
                "score": str(score),
                "observed_frequencies": observed_frequencies,
                "expected_frequencies": expected_frequencies,
                "observed_frequencies_norm": observed_frequencies_norm,
                "expected_frequencies_norm": expected_frequencies_norm,
                "p_value": str(p_value)
            })
    results = pd.DataFrame(results)
    return results

if __name__ == '__main__':

    datain = pd.read_csv(DATA_FILE)  # Uncomment to load as pandas.DataFrame.

    # Filter to data this year
    datain = datain.loc[datain["Date & Time"].str.contains("2022")]
    # data = data.replace('---', np.nan)

    dfs = []

    # Loop over available actions
    for action in datain["Action"].unique():

        # Filter to just one activity to make distributions cleaner
        data = datain.copy().loc[datain["Action"]==action]
        cols = list(data.columns[27:])

        if data.shape[0] < 100:
            continue

        # Remove Signature field
        cols.remove('Health facility or laboratory staff signature')

        # Remove numeric columns
        num_cols = data._get_numeric_data().columns
        cols = list(set(cols) - set(num_cols))

        # Subset to populated question columns for this 'Action'
        cols = [INTERVIEWER_ID] + cols
        data = data[cols]
        nunique = data.nunique()
        cols_to_drop = nunique[nunique == 1].index
        data.drop(cols_to_drop, axis=1, inplace=True)

        QUESTIONS = list(data.columns)
        QUESTIONS.remove(INTERVIEWER_ID)

        print("Analyzing the following columns ...")
        for c in data.columns:
            print(c)

        # data = mlab.csv2rec(DATA_FILE)  # Uncomment to load as numpy.recarray.

        # Compute MMA outlier scores.  Will work only if scipy is installed.
        if hasattr(outlierdetect, 'run_mma'):
            results = compute_mma(data, QUESTIONS, action)

        # results = compute_sva(data, QUESTIONS, action) # Uncomment to use the SVA algorithm.
        dfs.append(results)

    results = pd.concat(dfs)
    results = results.copy().loc[results["score"] != "nan"]
    results['score'] = pd.to_numeric(results['score'])
    results = results.sort_values(by='score', ascending=False)
    results.to_csv('./results.csv')
