from tqdm.autonotebook import tqdm
from deepface.commons import distance as dst
import pandas as pd
import numpy as np
import os

from sklearn.cluster import DBSCAN

models = [
    "SFace",
    "Facenet",
    "Facenet512",
    "VGG-Face",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
]

distance_metrics = [
    "cosine",
    "euclidean",
]

thresholds = np.arange(0.2, 0.32, 0.01)

for model_name in tqdm(models):
    save_path = f"lfw_people_benchmark/results_dbscan/{model_name}_results.csv"
    # load the corresponding pre-computed representation
    df_embed: pd.DataFrame = pd.read_pickle(
        f"lfw_people_benchmark/representation/{model_name}/{model_name}_representations.pkl"
    )

    for distance_metric in distance_metrics:
        threshold_base = dst.findThreshold(model_name=model_name, distance_metric=distance_metric)
        for threshold in [threshold_base]:
            for min_sample in range(2, 6):
                print(
                    f"{model_name} with {distance_metric}, {min_sample} min_samples, and {threshold} threshold",
                    flush=True,
                )
                df_res = df_embed.copy()

                clustering = DBSCAN(eps=threshold, min_samples=min_sample, metric=distance_metric)

                X = np.array(df_res[f"{model_name}_representation"].to_list())
                res = clustering.fit(X)
                df_res["cluster_label"] = res.labels_

                assert df_res["target"].nunique() == 5749  # 5749 unique people in the dataset

                # excluded the outliers
                df_res = df_res.groupby("target").filter(lambda x: len(x) > 1)

                # predited nb of clusters
                nb_clusters_predicted = df_res["cluster_label"].nunique()

                # average number of different people per cluster
                df_stats = (
                    df_res.groupby("cluster_label")
                    .agg(
                        {
                            "target": ["nunique", "count"],
                        }
                    )
                    .reset_index()
                )

                df_stats.columns = ["_".join(col) for col in df_stats.columns]
                people_per_cluster_wmean = (
                    df_stats["target_nunique"] * (df_stats["target_count"] / df_stats["target_count"].sum())
                ).sum()
                people_per_cluster_mean = df_stats["target_nunique"].mean()

                # "correctly" clustered people

                # for each group get the most frequent target
                df_res["most_frequent_target"] = df_res.groupby("cluster_label")["target"].transform(
                    lambda x: x.value_counts().index[0]
                )

                # check if the most frequent target is the same as the target
                df_res["is_correct"] = df_res["target"] == df_res["most_frequent_target"]

                # get the accuracy
                correctly_clustered_mean = df_res["is_correct"].mean()

                # distribution of the dominant identities
                df_most = (
                    df_res.groupby("cluster_label")["target"].agg(lambda x: x.value_counts().index[0]).reset_index()
                )
                identity_per_cluster_mean = df_most["target"].value_counts().mean()
                # identity_per_cluster_std = df_most["target"].value_counts().std()
                identity_per_cluster_max = df_most["target"].value_counts().max()

                # nb identities
                nb_identities = df_res["most_frequent_target"].nunique()

                # summary of the clustering
                summary = {
                    "model_name": model_name,
                    "distance_metric": distance_metric,
                    "threshold": threshold,
                    "min_samples": min_sample,
                    "#PC": nb_clusters_predicted,
                    "WAIC": people_per_cluster_wmean,
                    "id_per_cluster_mean": people_per_cluster_mean,
                    "DIAM": correctly_clustered_mean,
                    "nb_identities": nb_identities,
                    "dist_dominant_mean": identity_per_cluster_mean,
                    "dist_dominant_max": identity_per_cluster_max,
                }

                # transform res to a dataframe
                df_tosave = pd.DataFrame(summary, index=[0])

                # check if the file already exists

                if os.path.isfile(save_path):
                    df_results = pd.read_csv(save_path)
                    df_results = pd.concat([df_results, df_tosave])
                else:
                    df_results = df_tosave

                # save the results
                df_results.to_csv(save_path, index=False)
