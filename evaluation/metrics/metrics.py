from evaluation.metrics.intervals import get_interval_distances_table
from evaluation.metrics.musicality import get_information_rate_table
from evaluation.metrics.plagiarism import get_plagiarism_ranking_table


def obtain_metrics(df, e_orig, e_dest):
    return {"plagiarism": get_plagiarism_ranking_table(df),
            "intervals": get_interval_distances_table(df, e_orig, e_dest),
            "musicality": get_information_rate_table(df)
            }
