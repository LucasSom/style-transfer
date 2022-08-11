import numpy as np
from scipy.stats import entropy

from evaluation.metrics.intervals import get_style_avg
from evaluation.metrics.plagiarism import sort_by_general_plagiarism, get_most_similar_roll


def evaluate_model(df, metrics, column=None):
    print("===== Evaluate interval distributions =====")
    o, t = evaluate_transference(df, column)
    print("How many rolls moved away?", o)
    print("How many rolls moved closer to the new style?", t)


def evaluate_plagiarism_coincidences(df, direction) -> float:
    rolls = list(df['rolls'])
    base_rolls = df[direction]
    titles = list(df['Title'])

    similarities = [title == get_most_similar_roll(base_roll, rolls).song.name
                    for title, base_roll in zip(titles, base_rolls)]
    return sum(similarities) / len(similarities)


def evaluate_plagiarism_rate(df, direction) -> float:
    rolls = list(df['rolls'])
    titles = list(df['Title'])
    base_rolls = df[direction]

    distincts = 0
    for title, base_roll in zip(titles, base_rolls):
        sorted_rolls = sort_by_general_plagiarism(rolls, base_roll)
        for r in sorted_rolls:
            if r.song.name == title:
                break
            else:
                distincts += 1
    return distincts / len(rolls)


def evaluate_interval_distribution(m_orig, m_trans, orig_avg, trans_avg):
    def cmp_interval_matrices(m, avg):
        return np.mean([entropy(avg, m), entropy(m, avg)])

    oo = cmp_interval_matrices(m_orig, orig_avg)
    ot = cmp_interval_matrices(m_orig, trans_avg)
    to = cmp_interval_matrices(m_trans, orig_avg)
    tt = cmp_interval_matrices(m_trans, trans_avg)
    return oo < ot and oo < to, \
           tt > ot and tt > to


def evaluate_transference(df, column, orig=None, dest=None, eps=0.00001):
    """df_transferred"""
    orig_style_mx = get_style_avg(df, orig)
    trans_style_mx = get_style_avg(df, dest)
    orig_style_mx_norm = orig_style_mx + eps / np.sum(orig_style_mx + eps)
    trans_style_mx_norm = trans_style_mx + eps / np.sum(trans_style_mx + eps)
    improvements = [0, 0]

    for r_orig, r_trans in zip(df['roll'], df[column]):
        o, t = evaluate_interval_distribution(r_orig.matrix, r_trans.matrix, orig_style_mx_norm, trans_style_mx_norm)
        improvements[0] += o
        improvements[1] += t

    return improvements
