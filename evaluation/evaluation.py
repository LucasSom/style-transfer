from evaluation.metrics.intervals import get_interval_distances_table
from evaluation.metrics.plagiarism import sort_by_general_plagiarism, get_most_similar_roll


def evaluate_model(df, metrics, column=None):
    print("===== Evaluate interval distributions =====")
    o, t = get_interval_distances_table(df, column)
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


