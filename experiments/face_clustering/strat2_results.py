import pandas as pd
import pathlib
import sys
import numpy as np
from ast import literal_eval
from deepface.commons import distance as dst
from tqdm import tqdm

sys.path.append("../")
from helper.display_cluster import show_cluster, faceId_to_ogId

print("Loading the clustering results...", flush=True)

# Load the data
base_path = pathlib.Path("/media/bao/t7/la_lib_dataset")
src_folder = base_path / "img"
faces_folder = base_path / "faces"

# results FACENET512 (best f1 score)
res_facenet512 = base_path / "results_dbscan" / "df" / "cluster_Facenet512_DBSCAN_cosine_5_0.24.csv"
df_cluster_512 = pd.read_csv(res_facenet512, usecols={"image", "cluster_label"})

# results FACENET(128) (best f1 score)
res_facenet128 = base_path / "results_dbscan" / "df" / "cluster_Facenet_DBSCAN_cosine_2_0.22.csv"
df_cluster_128 = pd.read_csv(res_facenet128, usecols={"image", "cluster_label"})

print("Loading the embeddings...", flush=True)

# Load the embeddings
model_name = "Facenet512"
# df = pd.read_csv(df_folder / f"keep_representation_{model_name}.csv", index_col=0, converters={f"{model_name}_representation": literal_eval})
df_representation = pd.read_csv(
    "/media/bao/t7/la_lib_dataset/results_dbscan/df/" + f"keep_representation_{model_name}.csv",
    usecols={"image", f"{model_name}_representation"},
    index_col=0,
    converters={f"{model_name}_representation": literal_eval},
).reset_index()

# merge results on image
df = df_cluster_512.merge(df_cluster_128, on="image", suffixes=("_512", "_128"))

# left join on image only add Facenet512_representation to df
df = df.merge(df_representation[["image", f"{model_name}_representation"]], on="image", how="left")


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


df_avg_embed_512 = get_avg_embed(df, "Facenet512", "cluster_label_512")

# left join on cluster_label only add avg_Facenet512_representation to df
df = df.merge(df_avg_embed_512, left_on="cluster_label_512", right_on="cluster_label", how="left")


def calc_dst_to_mean(row):
    # check if the row is outlier
    if row["cluster_label_512"] == -1:
        return np.NaN
    return dst.findCosineDistance(row["avg_Facenet512_representation"], row["Facenet512_representation"])


df["dst_to_mean"] = df.apply(calc_dst_to_mean, axis=1)

print("Loading the original filenames...", flush=True)

# dataframe where the original details are stored from the API
df1 = pd.read_csv("/media/bao/t7/la_lib_dataset/df/df1.csv", converters={"metadata": literal_eval})
df2 = pd.read_csv("/media/bao/t7/la_lib_dataset/df/df2.csv", converters={"metadata": literal_eval})
df3 = pd.read_csv("/media/bao/t7/la_lib_dataset/df/df3.csv", converters={"metadata": literal_eval})
df4 = pd.read_csv("/media/bao/t7/la_lib_dataset/df/df4.csv", converters={"metadata": literal_eval})

df_metadata = pd.concat([df1, df2, df3, df4], ignore_index=True)

df_metadata["filename"] = df_metadata["metadata"].apply(lambda x: x.get("filename"))

df["id"] = df["image"].apply(faceId_to_ogId)

# left join on id only add filename to df
df = df.merge(df_metadata[["id", "filename"]], on="id", how="left")

# create a title column for the plots
df["title"] = df["filename"] + " \n " + df["dst_to_mean"].round(3).astype(str)

# groupy by cluster_label_512
df_grouped_512 = df.groupby("cluster_label_512")


def get_corresponding_cluster_128(x):
    val_counts = x["cluster_label_128"].value_counts().reset_index()
    # get first value != -1 and count < 100
    for i, row in val_counts.iterrows():
        cluster_id = row["index"]
        cluster_count = row["cluster_label_128"]
        if (cluster_id != -1) and (cluster_count < 100):
            return cluster_id
    return -1


df_matching_cluster_id = (
    df_grouped_512.apply(lambda x: get_corresponding_cluster_128(x))
    .reset_index()
    .rename(columns={0: "cluster_label_128"})
)

# left join df and df_matching_cluster_id
df = df.merge(df_matching_cluster_id, on="cluster_label_512", how="left")

# sort df by dst_to_mean
df = df.sort_values(by="dst_to_mean", ascending=True)

print("Generating the plots...", flush=True)

# groupby by cluster_label_512 and iterate over the groups
for idx, (cluster_id_512, group) in tqdm(enumerate(df.groupby("cluster_label_512")), total=len(df_grouped_512)):

    if cluster_id_512 not in [72, 109, 157, 175, 183, 228, 233, 234, 236, 261]:
        continue

    # ignore the outliers (-1)
    if cluster_id_512 == -1:
        continue

    limit = 100
    group_size = len(group)
    # ignore the clusters with more than 100 images
    if group_size > limit:
        continue

    # print(f"    cluster_id_512: {cluster_id_512}", flush=True)

    # get the cluster_id_128
    cluster_id_128 = group["cluster_label_128_y"].iloc[0]

    if cluster_id_128 != -1:
        ids_to_highlight, _ = show_cluster(
            df=group,
            cluster_id=cluster_id_128,
            faces_folder=faces_folder,
            originals_folder=src_folder,
            limit=limit,
            ncol=10,
            show_original=True,
            plot=False,
            save_folder=None,
            hide_axis=False,
            title_col=None,
            marked=[],
            clusterColName="cluster_label_128_x",
        )
    else:
        ids_to_highlight = []

    # suffix = 001, 002, 003, ...
    suffix = str(group_size).zfill(3) + "_"

    ids = show_cluster(
        df=group,
        cluster_id=cluster_id_512,
        faces_folder=faces_folder,
        originals_folder=src_folder,
        limit=limit,
        ncol=5,
        show_original=True,
        plot=True,
        save_folder=pathlib.Path("../my_cluster/40k-strat2"),  # pathlib.Path("../my_cluster/40k-strat2"),
        save_suffix=suffix,
        hide_axis=False,
        title_col="title",
        marked=ids_to_highlight,
        marked_color="green",
        clusterColName="cluster_label_512",
    )

    # if idx > 5:
    #     break

    # break
