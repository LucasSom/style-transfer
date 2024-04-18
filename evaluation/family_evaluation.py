import pandas as pd
import seaborn as sns
from typing import Dict, List

from utils.files_utils import load_pickle
from utils.plots_utils import save_plot


def get_family_metrics(eval_paths: Dict[str, List[str]], kind='joined'):
    """
    Returns a DataFrame with metrics for a particular family. The columns are 'mutation' and each evaluation category
    ('Style', 'Musicality' and 'Plagiarism-[dist|diff]').
    """
    d = []
    for mutation, files in eval_paths.items():
        dicts_overall_metrics = [load_pickle(f) for f in files]
        d.append(pd.DataFrame({
            "mutation": [mutation] * len(dicts_overall_metrics),
            "Style": [d["Style"][kind][d["target"]] for d in dicts_overall_metrics],
            "Musicality": [d["Musicality"] for d in dicts_overall_metrics],
            "Plagiarism-dist": [d["Plagiarism-dist"] for d in dicts_overall_metrics],
            "Plagiarism-diff": [d["Plagiarism-diff"] for d in dicts_overall_metrics],
        }))
    return pd.concat(d)


def evaluate_family_metrics(family_metrics, output_path, mutations_add, mutations_sub):
    sns.set_theme()
    sns.set_context("talk")

    family_metrics["alpha"] = family_metrics.apply(lambda row: row["mutation"].split("_")[-1], axis=1)

    for category in "Style", "Musicality", "Plagiarism-dist", "Plagiarism-diff":
        df_add = family_metrics[family_metrics["mutation"].isin(mutations_add)]
        sns.boxplot(x="alpha", y=category, data=df_add)
        save_plot(output_path, f"boxplot-{category}-add", subfolder=False)

        df_sub = family_metrics[family_metrics["mutation"].isin(mutations_sub)]
        sns.boxplot(x="alpha", y=category, data=df_sub)
        save_plot(output_path, f"boxplot-{category}-sub", subfolder=False)
