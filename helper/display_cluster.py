import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def faceId_to_ogId(faceId: str) -> str:
    """
    Convert faceId to original Id

    @param faceId: faceId of image, eg 0-dHzE7f4Yd8PuTtTAKwAN_f0
    @return: original Id of image, eg 0-dHzE7f4Yd8PuTtTAKwAN
    """
    ogId = faceId.split("_f")[:-1]
    ogId = "_f".join(ogId)  # just in case there are more _f in the name
    return ogId


def show_cluster(
    df: pd.DataFrame,
    cluster_id: int,
    faces_folder: pathlib.Path,
    originals_folder: pathlib.Path,
    limit=50,
    ncol=5,
    show_original=False,
    plot=True,
    save_folder: pathlib.Path = None,
    save_suffix: str = "",
    hide_axis=False,
    title_col: str = None,
    marked: list[str] = [],
    marked_color: str = "red",
    clusterColName: str = "cluster_label",
    disable_tqdm=True,
) -> list[str]:
    """
    Show cluster with cluster_id

    @param df: dataframe with at least columns: image (image name),
    cluster_label
    @param cluster_id: cluster id to show
    @param faces_folder: folder contains faces images
    @param originals_folder: folder contains original images
    @param limit: maximum number of images to show
    @param ncol: number of columns
    @param show_original: show original image or not (i.e. show face image)
    @param plot: plot the cluster or not
    @param save_folder: folder to save the plot
    @param save_suffix: suffix to add to the saved file
    @param hide_axis: hide axis or not
    @param title_col: column name to show as title
    @param marked: list of image id to mark with marked_color border
    @param marked_color: the color of the border of marked image
    @parem clusterColName: name of column containing cluster label

    @return ids: list of all ids in the cluster
    @return marked_ids: list of marked ids in the cluster
    """

    cluster = df[df[clusterColName] == cluster_id]
    cluster_size = cluster.shape[0]

    nrow = int(np.ceil(cluster_size / ncol))
    nb_max_nrow = int(np.ceil(limit / ncol))
    nrow = min(nrow, nb_max_nrow)

    nb_subplots = nrow * ncol

    if plot:
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 5))
        # fig.tight_layout()
        axs: list[plt.Axes] = axs.flatten()
        # add title to the figure
        # fig.suptitle(f"Cluster {cluster_id}: {cluster_size} images", fontsize=16)

    ids: list[str] = []
    marked_ids: list[str] = []

    # iterate over the cluster
    for i, (idx, row) in tqdm(enumerate(cluster.iterrows()), total=cluster.shape[0], disable=disable_tqdm):
        image: str = row["image"]  # face image name: eg. 0-dHzE7f4Yd8PuTtTAKwAN_f0.jpg
        img_src = faces_folder / image

        # reconstruct the original image path
        if show_original:
            img_suffix = img_src.suffix  # .jpg
            img_orig = img_src.stem  # 0-dHzE7f4Yd8PuTtTAKwAN_f0
            img_name = faceId_to_ogId(img_orig)  # 0-dHzE7f4Yd8PuTtTAKwAN
            img_src = originals_folder / f"{img_name}{img_suffix}"

        id = img_src.stem  # 0-dHzE7f4Yd8PuTtTAKwAN_f0 or 0-dHzE7f4Yd8PuTtTAKwAN
        id_original = faceId_to_ogId(id)  # 0-dHzE7f4Yd8PuTtTAKwAN

        ids.append(id)

        if (id in marked) or (id_original in marked):
            marked_ids.append(id)

        if not plot:
            continue

        # check if image exist
        if not img_src.exists():
            print(f"Image {img_src} does not exist")
            continue

        img = cv2.imread(str(img_src))

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(f"Image {img_src} is Corrupted (cluster {cluster_id})")
            continue

        axs[i].imshow(img)

        if (id in marked) or (id_original in marked):
            for pos in ["top", "bottom", "left", "right"]:
                axs[i].spines[pos].set_edgecolor(marked_color)
                axs[i].spines[pos].set_linewidth(5)

        # set title
        if title_col is not None:
            title = row[title_col]
        else:
            title = f"{i}: {img_src.name} - {img.shape[0]}x{img.shape[1]}"
        axs[i].set_title(title)
        if hide_axis and not (id in marked) and not (id_original in marked):
            axs[i].axis("off")

        if i >= limit:
            break

    # hide all the remaining axis
    if plot:
        for i in range(cluster_size, nb_subplots):
            axs[i].axis("off")

    if plot and save_folder:
        fig.savefig(save_folder / f"{save_suffix}cluster_{cluster_id}.png")
        plt.close(fig)

    return ids, marked_ids


def split_ids(ids: list[str], index_to_mark: list[int]):
    """
    Split ids into 2 parts: marked and unmarked

    @param ids: list of ids
    @param index_to_mark: list of index to mark

    @return: marked, unmarked
    """
    marked = [ids[i] for i in index_to_mark]
    unmarked = [ids[i] for i in range(len(ids)) if i not in index_to_mark]
    return marked, unmarked


def locate_and_plot_image(df: pd.DataFrame, image_id: str, clusterColName="cluster_label", **kwargs) -> list[str]:
    """
    Plot the cluster containing the image_id

    @param df: dataframe with at columns: image (image name), cluster_label
    @param image_id: the id of the image
    @param kwargs: arguments for show_cluster

    @return: list of image id in the cluster
    """
    show_id = df[df["image"].str.startswith(image_id)][clusterColName].values[0]

    if show_id == -1:
        print(f"Image {image_id} is not in any cluster")
        return []

    ids = show_cluster(df=df, cluster_id=show_id, clusterColName=clusterColName, **kwargs)
    return ids


def plot_overview_cluster(
    df: pd.DataFrame, face_folder: pathlib.Path, offset=0, nrow=10, ncol=11, clusterColName: str = "cluster_label"
) -> None:

    nb_cluster_end = nrow + offset
    cluster_ids = df[clusterColName].value_counts().index[offset:nb_cluster_end]

    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 3))
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)

    for idx, cluster_id in enumerate(cluster_ids):

        length = df[df[clusterColName] == cluster_id].shape[0]
        cluster = df[df[clusterColName] == cluster_id][: (ncol - 1)]
        legend = f"Cluster {cluster_id}\n{length} images"
        axs[idx, 0].text(0.1, 0.5, legend, fontsize=14, color="red")
        axs[idx, 0].axis("off")

        for j, image in enumerate(cluster["image"]):

            img_src = face_folder / image
            # check if image exist
            if img_src.exists():
                img = cv2.imread(str(img_src))
            else:
                print(f"Image {img_src} not found")
                continue

            # check if image is None
            if img is None:
                print(f"Image {img_src} is corrupted")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[idx, j + 1].imshow(img)
            axs[idx, j + 1].axis("off")
            # ax[idx, j].set_title(img_path)

        # hide all the remaining axis
        for j in range(length, ncol - 1):
            axs[idx, j + 1].axis("off")
