from evaluation.metrics.intervals import get_interval_distances_table


def obtain_metrics(df, e_orig, e_dest, column):
    return get_interval_distances_table(df, e_orig, e_dest)

