from deepface.commons import distance as dst

from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np
import os

from sklearn.cluster import AgglomerativeClustering

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
    # "euclidean_l2", # not supported by default in sklearn.cluster.AgglomerativeClustering
]

linkages = [
    "ward",
    "complete",
    "average",
    "single",
]

for model_name in tqdm(models):
    save_path = f"lfw_people_benchmark/results_ahc_no_outliers/{model_name}_results.csv"
    # load the corresponding pre-computed representation
    df_embed = pd.read_pickle(f"lfw_people_benchmark/representation/{model_name}/{model_name}_representations.pkl")

    for distance_metric in distance_metrics:
        threshold = dst.findThreshold(model_name=model_name, distance_metric=distance_metric)

        affinity = "l2" if distance_metric == "euclidean_l2" else distance_metric

        for linkage in linkages:
            if linkage == "ward" and affinity != "euclidean":
                continue

            print(f"{model_name} with {distance_metric} and {linkage} linkage", flush=True)
            df_res = df_embed.copy()

            clustering = AgglomerativeClustering(
                distance_threshold=threshold, affinity=affinity, linkage=linkage, n_clusters=None
            )

            X = np.array(df_res[f"{model_name}_representation"].to_list())
            res = clustering.fit(X)
            df_res["cluster_label"] = res.labels_

            # assert df_res["target"].nunique() == 5749  # 5749 unique people in the dataset

            # excluded the outliers: to compare with DBSCAN
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
            df_most = df_res.groupby("cluster_label")["target"].agg(lambda x: x.value_counts().index[0]).reset_index()
            identity_per_cluster_mean = df_most["target"].value_counts().mean()
            # identity_per_cluster_std = df_most["target"].value_counts().std()
            identity_per_cluster_max = df_most["target"].value_counts().max()

            # nb identities
            nb_identities = df_res["most_frequent_target"].nunique()

            # summary of the clustering
            summary = {
                "model_name": model_name,
                "affinity": affinity,
                "threshold": threshold,
                "linkage": linkage,
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

            # save df_res
            # clustered_folder = pathlib.Path(f"lfw_people_benchmark/clustered/{model_name}")
            # # create the folder if it does not exist
            # clustered_folder.mkdir(parents=True, exist_ok=True)
            # df_res[["target", "cluster_label"]].to_csv(
            #     clustered_folder / f"{model_name}_{distance_metric}_{linkage}.csv"
            # )

            # df_res.to_pickle(clustered_folder / f"{model_name}_{distance_metric}_{linkage}_clustered.pkl")
