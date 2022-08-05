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
