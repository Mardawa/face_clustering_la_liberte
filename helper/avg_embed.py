import pandas as pd
import numpy as np


def get_avg_embed(df: pd.DataFrame, model_name: str, clusterColName: str = "cluster_label") -> pd.DataFrame:
    df_avg_embed = pd.DataFrame(columns=[f"cluster_label", f"avg_{model_name}_representation"])

    # group by cluster_id
    for cluster_id, group in df.groupby(clusterColName):
        # ignore the outliers (-1)
        if cluster_id == -1:
            continue

        cluster_name = cluster_id

        matrix = np.array(group[f"{model_name}_representation"].tolist())
        avg_embed = np.mean(matrix, axis=0)
        df_avg_embed = pd.concat(
            [
                df_avg_embed,
                pd.DataFrame(
                    [[avg_embed, cluster_name]], columns=[f"avg_{model_name}_representation", "cluster_label"]
                ),
            ],
            ignore_index=True,
        )

    return df_avg_embed
