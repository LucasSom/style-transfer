import os
import numpy as np
from scipy.stats import entropy


def filter_column(df, column="Embedding", tipo='Fragmento'):
    return {
        nombre: roll
        for nombre, roll, t in zip(df['Title'], df[column], df['Tipo'])
        if t == tipo
    }


# Cuales son los exps que hicimos?
def exp_disponibles(df):
    c_no_exp = {'Style', 'Title', 'roll', 'oldPM', 'Tipo', 'Sigma'}
    return [c for c in df.columns
            if c not in c_no_exp
            and 'roll' not in c
            and 'midi' not in c]


def normalize(m, eps=0.00001):
    m_sum = np.sum(m + eps)
    return m + eps / m_sum


def show_sheets(df_transferred, column, sheets_path, suffix):
    titles = (df_transferred['Title'] if suffix is None
              else df_transferred['Title'].map(lambda t: f'{t}_{suffix}'))
    rolls = df_transferred[column]
    for title, roll in zip(titles, rolls):
        sheet_path = os.path.join(sheets_path, title)
        roll.display_score(file_name=sheet_path, fmt='png', do_display=False)


def get_matrix_comparisons(m_orig, m_trans, orig_avg, trans_avg):
    """
    :param m_orig: matrix of the original roll
    :param m_trans: matrix of the transformed roll
    :param orig_avg: matrix from the original style
    :param trans_avg: matrix from the target style
    """
    return {
        "ms": cmp_matrices(m_orig, orig_avg),
        "ms'": cmp_matrices(m_orig, trans_avg),
        "m's": cmp_matrices(m_trans, orig_avg),
        "m's'": cmp_matrices(m_trans, trans_avg)
    }


def cmp_matrices(m, avg):
    assert m.shape == avg.shape
    m_normalized = normalize(m)
    return np.mean([entropy(avg, m_normalized), entropy(m_normalized, avg)])
