import pandas as pd
import seaborn as sns
from typing import Dict, List

from utils.files_utils import load_pickle
from utils.plots_utils import save_plot


def get_family_metrics(eval_paths: Dict[str, List[str]]):
    """
    Returns a DataFrame with metrics for a particular family. The columns are 'mutation' and each evaluation category
    ('Style', 'Musicality' and 'Plagiarism-[dist|diff]').
    """
    d = []
    for mutation, files in eval_paths.items():
        for kind in 'melodic', 'rhythmic', 'joined':
            dicts_overall_metrics = [load_pickle(f) for f in files]
            d.append(pd.DataFrame({
                "mutation": [mutation] * len(dicts_overall_metrics),
                "kind": [kind] * len(dicts_overall_metrics),
                "Style": [d["Style"][kind][d["target"]] for d in dicts_overall_metrics],
                "Musicality": [d["Musicality"] for d in dicts_overall_metrics],
                "Plagiarism-dist": [d["Plagiarism-dist"] for d in dicts_overall_metrics],
                "Plagiarism-diff": [d["Plagiarism-diff"] for d in dicts_overall_metrics],
            }))
    return pd.concat(d)


def evaluate_family_metrics(family_metrics, output_path, mutations_add, mutations_sub, individual=True):
    sns.set_theme(context="talk", style="whitegrid")

    family_metrics["alpha"] = family_metrics.apply(lambda row: row["mutation"].split("_")[-1], axis=1)

    for category in "Style", "Musicality", "Plagiarism-dist", "Plagiarism-diff":
        # df_add = family_metrics[family_metrics["mutation"].isin(mutations_add)]
        df_sub = family_metrics[family_metrics["mutation"].isin(mutations_sub)]

        if individual:
            for kind in 'melodic', 'rhythmic', 'joined':
                # sns.boxplot(x="alpha", y=category, data=df_add[df_add["kind"] == kind])
                # save_plot(output_path, f"boxplot-{category}-{kind}-add", subfolder=False)

                sns.boxplot(x="alpha", y=category, data=df_sub[df_sub["kind"] == kind])
                save_plot(output_path, f"boxplot-{category}-{kind}-sub", subfolder=False)

        else:
            # sns.boxplot(data=df_add, x="kind", y=category, hue="alpha")
            # save_plot(output_path, f"boxplot-{category}-add", subfolder=False)

            sns.boxplot(data=df_sub, x="kind", y=category, hue="alpha")
            save_plot(output_path, f"boxplot-{category}-sub", subfolder=False)

def get_families_metrics(family_paths: List[str], models_alias):
    dfs = []
    for p in family_paths:
        model_alias = models_alias[p.split('/')[-2]]
        df = load_pickle(p)
        df = df[df['kind'] == 'joined']
        df['model'] = 36 * [model_alias]
        dfs.append(df)
    return pd.concat(dfs)


def evaluate_families_metrics(df, output_dir):
    df.rename(columns={"Plagiarism-dist": "Similarity"}, inplace=True)
    # sns.set(rc={'axes.facecolor': 'white'})
    sns.set_style('whitegrid')
    for metric in 'Musicality', 'Style', 'Similarity':
        sns.boxplot(data=df, x='alpha', y=metric, hue='model', palette={'pre': 'white', 'post': 'grey'})
        save_plot(output_dir, metric)
