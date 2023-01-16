import os
import numpy as np


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


def normalize(m):
    m_sum = np.sum(m + eps)
    return m + eps / m_sum


eps = 0.0001


def show_sheets(df_transferred, column, sheets_path, suffix):
    titles = (df_transferred['Title'] if suffix is None
              else df_transferred['Title'].map(lambda t: f'{t}_{suffix}'))
    rolls = df_transferred[column]
    for title, roll in zip(titles, rolls):
        sheet_path = os.path.join(sheets_path, title)
        roll.display_score(file_name=sheet_path, fmt='png', do_display=False)
