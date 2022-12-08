import pathlib
import pandas as pd
import json
import sys

sys.path.append("../")
from helper.display_cluster import show_cluster


def get_cluster_ids(df, face_id):
    cluster = df[df["face_id"] == face_id]["cluster_label"].values[0]
    return cluster


def get_TP_FN_FP(
    ids: list[str],
    df: pd.DataFrame,
    faces_folder: pathlib.Path,
    src_folder: pathlib.Path,
    clusterColName: str = "cluster_label",
    plot=False,
):
    """
    Compute the true positive and false negative based on a list of ids

    Args:
        ids (list[str]): list of ids
        df (pd.DataFrame): dataframe containing the cluster labels

    Returns:
        dominant_cluster (str): dominant cluster
        tp (int): number of true positive (how many did we get right)
        fn (int): number of false negative (how many did we miss)
        fp (int): number of false positive (how many did we get wrong)
    """
    df_ids = pd.DataFrame(ids, columns=["image_id"])
    df_ids["cluster_id"] = df_ids["image_id"].apply(lambda x: get_cluster_ids(df, x))
    val_counts = (
        df_ids["cluster_id"].value_counts().reset_index().rename(columns={"index": "cluster_id", "cluster_id": "count"})
    )

    dominant_cluster = val_counts.iloc[0]["cluster_id"]
    if (dominant_cluster == -1) and (val_counts.shape[0] > 1):  # use the second most dominant cluster
        dominant_cluster = val_counts.iloc[1]["cluster_id"]
        tp = val_counts.iloc[1]["count"]
        fn = val_counts.iloc[:]["count"].sum() - tp
    else:  # use the first most dominant cluster
        tp = val_counts.iloc[0]["count"]
        fn = val_counts.iloc[1:]["count"].sum()

    ids_dom_clust, _ = show_cluster(
        df=df,
        cluster_id=dominant_cluster,
        faces_folder=faces_folder,
        originals_folder=src_folder,
        limit=50,
        ncol=10,
        show_original=False,
        plot=False,
        save_folder=None,
        hide_axis=False,
        title_col=None,
        marked=[],
    )

    # count the diff between ids_dom_clust and ids
    diff = list(set(ids_dom_clust) - set(ids))
    fp = len(diff)

    if plot:
        print("cluster", dominant_cluster)
        _ = show_cluster(
            df=df,
            cluster_id=dominant_cluster,
            faces_folder=faces_folder,
            originals_folder=src_folder,
            limit=50,
            ncol=10,
            show_original=True,
            plot=True,
            save_folder=None,
            hide_axis=False,
            title_col=None,
            marked=diff,
        )

    return dominant_cluster, tp, fn, fp


def compare_w_ref(
    reference_clusters_path: pathlib.Path, df: pd.DataFrame, faces_folder: pathlib.Path, src_folder: pathlib.Path
):
    df_stats = pd.DataFrame(
        columns=["cluster_ref_id", "n_images", "dominant_cluster", "tp", "fn", "fp", "precision", "recall", "f1"]
    )

    for file in reference_clusters_path.glob("*.json"):
        with open(file, "r") as f:
            ids = json.load(f)

        # performance stats
        dominant_cluster, tp, fn, fp = get_TP_FN_FP(
            ids, df, plot=False, faces_folder=faces_folder, src_folder=src_folder
        )

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        # general stats
        cluster_id = file.stem.split("_")[-1]
        length = len(ids)
        df_stats = pd.concat(
            [
                df_stats,
                pd.DataFrame(
                    [[cluster_id, length, dominant_cluster, tp, fn, fp, precision, recall, f1]],
                    columns=[
                        "cluster_ref_id",
                        "n_images",
                        "dominant_cluster",
                        "tp",
                        "fn",
                        "fp",
                        "precision",
                        "recall",
                        "f1",
                    ],
                ),
            ]
        )
        # break

    total_tp = df_stats["tp"].sum()
    total_fn = df_stats["fn"].sum()
    total_fp = df_stats["fp"].sum()

    # total_precision = total_tp / (total_tp + total_fp)
    # total_recall = total_tp / (total_tp + total_fn)
    # total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)

    # print(f"Total TP: {total_tp}, Total FN: {total_fn}, Total FP: {total_fp}")
    # print(f"Total precision: {total_precision}, Total recall: {total_recall},
    # Total F1: {total_f1}")

    return total_tp, total_fn, total_fp, df_stats
