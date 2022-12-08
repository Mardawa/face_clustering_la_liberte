import pathlib
import pandas as pd
from time import perf_counter

from deepface.commons import distance as dst

import sys

sys.path.append("../")

from face_clustering_pipeline import FaceClusteringPipeline
from helper.avg_embed import get_avg_embed


def find_closest_cluster(src_embed, df_avg_embed: pd.DataFrame, threshold=0.24) -> int:
    """
    Find the closest cluster to the given embedding with the given threshold
    """
    # make a copy of df_avg_embed
    df = df_avg_embed.copy()

    df["distance"] = df["avg_Facenet512_representation"].apply(lambda x: dst.findCosineDistance(src_embed, x))

    # find the min distance
    min_distance = df["distance"].min()

    if min_distance < threshold:
        # get the cluster label
        cluster_label = df[df["distance"] == min_distance]["cluster_label"].values[0]
    else:
        cluster_label = -1

    return cluster_label


def clustering_1by1(
    df_baseline: pd.DataFrame,
    df_1by1: pd.DataFrame,
    cluster_outlier: str,
    faceClusteringPipeline: FaceClusteringPipeline,
    df_folder: pathlib.Path,
    model_name: str = "Facenet512",
) -> list:
    """
    Clustering the images in df_1by1 one by one

    To test the performance of the clustering algorithm, we re-do the clustering
    for the baseline however, in pratice, it should be already computed.

    Args:
        df_baseline (pd.DataFrame): the baseline dataframe
        df_1by1 (pd.DataFrame): the dataframe containing the images to be
        clustered one by one
        cluster_outlier (str): either "new" or "all" or "skip". If "new", then only
        cluster the outliers from df_1by1. If "all", then cluster all the
        outliers, i.e. from both df_baseline and df_1by1. if "skip", then skip
        this step.

    Returns:
        pd.DataFrame: a dataframe with the clustering results
    """

    assert cluster_outlier in ["new", "all", "skip"], "cluster_outlier must be either 'new', 'existing' or 'skip'"

    # cluster the baseline
    df_baseline_res = faceClusteringPipeline.p_cluster_faces(
        df_baseline,
        df_folder,
        model_name=model_name,
        clustering_algo="DBSCAN",
        distance_metric="cosine",
        min_samples=5,
        threshold=0.24,
        save=False,
    )

    # compute the average embedding for each cluster
    df_avg_embed = get_avg_embed(df=df_baseline_res, model_name=model_name, clusterColName="cluster_label")

    t0 = perf_counter()
    df_1by1["cluster_label"] = df_1by1[f"{model_name}_representation"].apply(
        lambda x: find_closest_cluster(x, df_avg_embed)
    )

    if cluster_outlier == "new":
        # cluster only the outliers from df_1by1
        df_outliers = df_1by1[df_1by1["cluster_label"] == -1].copy()

        df_outliers_res = faceClusteringPipeline.p_cluster_faces(
            df_outliers,
            df_folder,
            model_name=model_name,
            clustering_algo="DBSCAN",
            distance_metric="cosine",
            min_samples=5,
            threshold=0.24,
            save=False,
        )

        max_cluster_label = df_1by1["cluster_label"].max() + 1
        df_outliers_res["cluster_label"] = df_outliers_res["cluster_label"].apply(
            lambda x: x + max_cluster_label if x != -1 else x
        )

        # replace the cluster label of the outliers
        df_1by1.loc[df_1by1["cluster_label"] == -1, "cluster_label"] = df_outliers_res["cluster_label"]

    df_res = pd.concat([df_baseline_res, df_1by1], ignore_index=True)

    # This column is needed to compare with the reference clusters
    df_res["face_id"] = df_res["image"].apply(pathlib.Path).apply(lambda x: x.stem)

    if cluster_outlier == "skip" or cluster_outlier == "new":
        t1 = perf_counter()
        t = t1 - t0
        return df_res, t

    # redondant with the previous if statement but it's clearer
    if cluster_outlier != "all":
        t1 = perf_counter()
        t = t1 - t0
        return df_res, t

    # cluster all the outliers
    df_outliers = df_res[df_res["cluster_label"] == -1].copy()

    df_outliers_res = faceClusteringPipeline.p_cluster_faces(
        df_outliers,
        df_folder,
        model_name=model_name,
        clustering_algo="DBSCAN",
        distance_metric="cosine",
        min_samples=5,
        threshold=0.24,
        save=False,
    )

    # change the cluster label of the outliers to the max cluster label + cluster label
    # so that the outliers are not mixed with the baseline
    # +1 because the cluster label starts from 0
    # ignore the -1 cluster label
    max_cluster_label = df_res["cluster_label"].max() + 1
    df_outliers_res["cluster_label"] = df_outliers_res["cluster_label"].apply(
        lambda x: x + max_cluster_label if x != -1 else x
    )

    # replace the cluster label of the outliers
    df_res.loc[df_res["cluster_label"] == -1, "cluster_label"] = df_outliers_res["cluster_label"]

    t1 = perf_counter()
    t = t1 - t0
    return df_res, t


def merge_clustering(
    clustering_res,
    df_folder: pathlib.Path,
    faceClusteringPipeline: FaceClusteringPipeline,
    model_name: str = "Facenet512",
):
    """
    Merge the clustering results of each part
    """
    # for each part, compute the average embedding of each cluster
    avg_embeds = []
    for idx, df_res in enumerate(clustering_res):
        avg_embed = get_avg_embed(df_res, model_name)
        avg_embed["subset"] = idx
        avg_embeds.append(avg_embed)

    # concatenate the average embeddings of each part
    df_avg_embeds = pd.concat(avg_embeds)

    # if df_avg_embeds is empty, return original df
    if df_avg_embeds.empty:
        return pd.concat(clustering_res)

    # rename the columns to avoid conflict
    df_avg_embeds = df_avg_embeds.rename(columns={"cluster_label": "cluster_label_src"})

    linkage = "average"
    threshold = 0.20

    df_AHC = faceClusteringPipeline.p_cluster_faces(
        df_avg_embeds,
        df_folder,
        model_name=model_name,
        clustering_algo="AHC",
        affinity="cosine",
        linkage=linkage,
        threshold=threshold,
        clusterColName="avg_Facenet512_representation",
        save=False,
    )

    res = []
    for idx, df_res in enumerate(clustering_res):

        df_final_cluster = df_AHC[df_AHC["subset"] == idx][["cluster_label", "cluster_label_src"]]

        df_res = df_res.rename(columns={"cluster_label": "cluster_label_src"})

        df_res = df_res.merge(df_final_cluster, on="cluster_label_src", how="left")

        res.append(df_res)

    df_res = pd.concat(res)
    df_res["cluster_label"] = df_res["cluster_label"].fillna(-1)
    df_res["cluster_label"] = df_res["cluster_label"].astype(int)
    df_res = df_res.drop(columns=["cluster_label_src"], errors="ignore")

    return df_res


def split_clustering(
    df_splits: list,
    faceClusteringPipeline: FaceClusteringPipeline,
    df_folder: pathlib.Path,
    model_name: str = "Facenet512",
):
    """
    Clustering each part and merge the results
    """
    # Clustering each part
    # best f1 : 5 0.24
    # best pre: 3 0.1925
    min_samples = 5
    eps = 0.24  # threshold

    clustering_res = []
    t_res = []

    for df_split in df_splits:
        t0 = perf_counter()
        df_res = faceClusteringPipeline.p_cluster_faces(
            df_split,
            df_folder,
            model_name=model_name,
            clustering_algo="DBSCAN",
            distance_metric="cosine",
            min_samples=min_samples,
            threshold=eps,
            save=False,
        )
        clustering_res.append(df_res)
        t1 = perf_counter()
        t_res.append(t1 - t0)

    t0 = perf_counter()
    df_res = merge_clustering(clustering_res, df_folder, faceClusteringPipeline, model_name=model_name)
    t1 = perf_counter()
    # t_res[1] += t1-t0
    t_res.append(t1 - t0)

    return df_res, t_res


def divide_clustering(
    df_splits: list,
    faceClusteringPipeline: FaceClusteringPipeline,
    df_folder: pathlib.Path,
    model_name: str = "Facenet512",
):
    """
    Clustering each part and merge the results
    """
    # Clustering each part
    # best f1 : 5 0.24
    # best pre: 3 0.1925
    min_samples = 5
    eps = 0.24  # threshold

    clustering_res = []
    t_res = []

    for df_split in df_splits:
        t0 = perf_counter()
        df_res = faceClusteringPipeline.p_cluster_faces(
            df_split,
            df_folder,
            model_name=model_name,
            clustering_algo="DBSCAN",
            distance_metric="cosine",
            min_samples=min_samples,
            threshold=eps,
            save=False,
        )
        clustering_res.append(df_res)
        t1 = perf_counter()
        t_res.append(t1 - t0)

    t0 = perf_counter()
    df_res = merge_clustering(clustering_res, df_folder, faceClusteringPipeline, model_name=model_name)
    t1 = perf_counter()
    t_res.append(t1 - t0)

    return df_res, t_res


def new_and_outliers_clustering(
    df_splits: list,
    faceClusteringPipeline: FaceClusteringPipeline,
    df_folder: pathlib.Path,
    model_name: str = "Facenet512",
):
    """
    Cluster [0] => baseline
    Cluster outliers of baseline + [1]
    Merge the results
    """
    # Clustering each part
    # best f1 : 5 0.24
    # best pre: 3 0.1925
    min_samples = 5
    eps = 0.24  # threshold

    clustering_res = []
    t_res = []

    # cluster the baseline
    t0 = perf_counter()
    df_res = faceClusteringPipeline.p_cluster_faces(
        df_splits[0],
        df_folder,
        model_name=model_name,
        clustering_algo="DBSCAN",
        distance_metric="cosine",
        min_samples=min_samples,
        threshold=eps,
        save=False,
    )
    # append the inliers of baseline
    clustering_res.append(df_res[df_res["cluster_label"] != -1].copy())
    t1 = perf_counter()
    t_res.append(t1 - t0)

    # cluster the outliers of baseline + [1]
    t0 = perf_counter()
    df_outliers = df_res[df_res["cluster_label"] == -1][["image", f"{model_name}_representation"]].copy()
    df = pd.concat([df_outliers, df_splits[1]])

    df_res = faceClusteringPipeline.p_cluster_faces(
        df,
        df_folder,
        model_name=model_name,
        clustering_algo="DBSCAN",
        distance_metric="cosine",
        min_samples=min_samples,
        threshold=eps,
        save=False,
    )

    clustering_res.append(df_res)

    # merge the results
    df_res = merge_clustering(clustering_res, df_folder, faceClusteringPipeline, model_name=model_name)
    t1 = perf_counter()
    t_res.append(t1 - t0)

    return df_res, t_res
