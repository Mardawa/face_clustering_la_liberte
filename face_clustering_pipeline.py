from time import perf_counter, time

import pandas as pd
import pathlib
import logging
import cv2

from deepface.commons import distance as dst
import numpy as np
import json


class FaceClusteringPipeline:
    """
    Pipeline to extract faces from images and cluster them

    Parameters
    ----------
    src_folder : path to folder containing images
    dst_folder : path to folder where to save extracted faces
    df_folder : path to folder where to save dataframes containing results
    log_folder : path to folder where to save logs
    """

    def __init__(
        self, src_folder: pathlib.Path, dst_folder: pathlib.Path, df_folder: pathlib.Path, log_folder: pathlib.Path
    ):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.df_folder = df_folder

        logfile = str(log_folder / f"{src_folder.name}_{time()}.log")

        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(message)s",
        )

        # creating necessary folders
        logging.info("Creating necessary folders")
        self.dst_folder.mkdir(parents=True, exist_ok=True)
        self.df_folder.mkdir(parents=True, exist_ok=True)

        # get number of images
        num_images = len(list(src_folder.glob("*")))
        logging.info(f"Number of file in src folder: {num_images}\n")

    def p_rotate_images(self, src_folder: pathlib.Path):
        from helper.rotate_image import rotate_images

        # rotate images
        logging.info("Rotating images")
        t0 = perf_counter()
        rotate_images(str(src_folder))
        t1 = perf_counter()
        logging.info(f"Rotating images took {t1 - t0} seconds\n")

    def p_extract_faces(
        self, src_folder: pathlib.Path, dst_folder: pathlib.Path, df_folder: pathlib.Path, threshold: float
    ) -> pd.DataFrame:

        from helper.face_detector import retinaface_detect_face_in_folder

        logging.info("Extracting faces")
        logging.info("  Model: RetinaFace")
        logging.info(f"  Threshold: {threshold}")
        t0 = perf_counter()
        res = retinaface_detect_face_in_folder(str(src_folder), str(dst_folder), threshold=threshold)
        t1 = perf_counter()
        logging.info(f"Extracting faces took {t1 - t0} seconds")

        # log results
        logging.info(f" {res['no_face']=}")
        logging.info(f" {res['not_image']=}")
        logging.info(f" {res['total_faces']=}\n")

        # save face detection results
        df_confidences = (
            pd.DataFrame.from_dict(res["confidences"], orient="index", columns=["confidence"])
            .reset_index()
            .rename(columns={"index": "image"})
        )
        df_confidences.to_csv(df_folder / "confidences.csv", index=False)

        return df_confidences

    def p_compute_blur(self, df: pd.DataFrame) -> pd.DataFrame:

        from blur_detection.blur_detection import detect_blur_laplacian, detect_blur_fft

        logging.info("Computing blur metrics")

        t0 = perf_counter()
        # compute laplacian
        logging.info("  Computing laplacian variance")
        df[["fm", "is_blurry_laplacian"]] = df.apply(
            lambda x: detect_blur_laplacian(x["img_bgr"], threshold=165, rescale=True), axis=1, result_type="expand"
        )
        t1 = perf_counter()
        logging.info(f"  Computing laplacian variance took {t1 - t0} seconds")

        t0 = perf_counter()
        # compute fft
        logging.info("  Computing fft")
        df[["fft", "is_blurry_fft"]] = df.apply(lambda x: detect_blur_fft(x["img_bgr"]), axis=1, result_type="expand")
        t1 = perf_counter()
        logging.info(f"  Computing fft took {t1 - t0} seconds")

        # get image size
        df["img_size"] = df.apply(lambda x: x["img_bgr"].shape, axis=1)

        return df

    def p_filter_faces(self, df: pd.DataFrame, df_folder: pathlib.Path, computeMetric: bool = True) -> pd.DataFrame:

        # compute blur metrics
        # Can filter out small images before to save time
        # Didn't do it here because it's not necessary for the demo
        # + the threshold could be changed
        if computeMetric:
            df = self.p_compute_blur(df)
        else:
            logging.info("Blur metrics already computed")

        # filter faces
        logging.info("Filtering faces")
        t0 = perf_counter()

        def keep_row(row, minimal_size=40, keep_size=250) -> bool:

            # From training on kaggle dataset
            ftt_threshold_high = 15.579238414764404
            laplacian_threshold_high = 165.1521759033203

            # Default suggested values
            ftt_threshold_low = 10
            laplacian_threshold_low = 100

            img_size_x = row["img_size"][0]
            img_size_y = row["img_size"][1]

            fm_score = row["fm"]  # laplacian
            fft_score = row["fft"]  # fft

            confidence = row["confidence"]  # confience score from face detector

            # Not necessary, since we set the threshold in the face detector (Retinaface) at
            # 0.95 already
            if confidence < 0.95:
                return False

            # discard image too small => discard image
            if img_size_x < minimal_size or img_size_y < minimal_size:
                return False

            # pass both low threshold => keep image
            if (fm_score > laplacian_threshold_low) and (fft_score > ftt_threshold_low):
                return True

            # pass at least 1 high threshold => keep image
            if (fm_score > laplacian_threshold_high) or (fft_score > ftt_threshold_high):
                return True

            # for high resolution images, lower the threshold
            if img_size_x > keep_size and img_size_y > keep_size:
                if fm_score > 10 and fft_score > 0.1:
                    return True

            # discard image
            return False

        df["keep"] = df.apply(lambda x: keep_row(x, minimal_size=40, keep_size=250), axis=1)
        t1 = perf_counter()
        logging.info(f"Filtering faces took {t1 - t0} seconds")

        nb_kept = df["keep"].sum()
        nb_kept_percent = df["keep"].mean()
        logging.info(f" {nb_kept} faces kept ({nb_kept_percent:.2%})\n")

        df[["image", "confidence", "fm", "fft", "img_size", "keep"]].to_csv(df_folder / "keep.csv", index=False)

        return df

    def p_represent_faces(self, df: pd.DataFrame, df_folder: pathlib.Path, model_name="Facenet512") -> pd.DataFrame:

        from deepface import DeepFace
        from deepface.basemodels import (
            VGGFace,
            Facenet,
            OpenFace,
            FbDeepFace,
            ArcFace,
            Facenet512,
            DeepID,
            DlibWrapper,
            SFaceWrapper,
        )

        logging.info(f"Representing faces using {model_name}")

        models = {
            "SFace": SFaceWrapper.load_model,
            "Facenet": Facenet.loadModel,
            "Facenet512": Facenet512.loadModel,
            "VGG-Face": VGGFace.loadModel,
            "OpenFace": OpenFace.loadModel,
            "DeepFace": FbDeepFace.loadModel,
            "DeepID": DeepID.loadModel,
            "Dlib": DlibWrapper.loadModel,
            "ArcFace": ArcFace.loadModel,
        }

        model = models[model_name]()

        t0 = perf_counter()
        # represent the images
        df[f"{model_name}_representation"] = df["img_bgr"].apply(
            lambda x: DeepFace.represent(
                img_path=x, model_name=model_name, model=model, enforce_detection=False, detector_backend="skip"
            )
        )
        t1 = perf_counter()
        logging.info(f"Representing faces took {t1 - t0} seconds\n")

        df[["image", "confidence", "fm", "fft", "img_size", f"{model_name}_representation"]].to_csv(
            df_folder / f"keep_representation_{model_name}.csv"
        )

        return df

    def p_cluster_faces(
        self,
        df: pd.DataFrame,
        df_folder: pathlib.Path,
        model_name="Facenet512",
        clustering_algo="AHC",
        save: bool = True,
        save_name: str = None,
        **kwargs,
    ) -> pd.DataFrame:

        logging.info("Clustering faces")
        n = df.shape[0]
        logging.info(f"  {n} faces to cluster")

        if clustering_algo == "AHC":
            df = self.p_cluster_faces_AHC(df, model_name=model_name, **kwargs)
        elif clustering_algo == "DBSCAN":
            df = self.p_cluster_faces_DBSCAN(df, model_name=model_name, **kwargs)

        if save_name is None:
            save_name = (
                str(df_folder)
                + f"/cluster_{model_name}_{clustering_algo}_"
                + "_".join([str(val) for val in kwargs.values()])
                + ".csv"
            )

        if save:
            logging.info(f"saving under {save_name}")
            df[["image", "confidence", "fm", "fft", "cluster_label"]].to_csv(save_name)

        # df[["image", "confidence", "fm", "fft", "img_size", f"{model_name}_representation", "cluster_label"]].to_csv(
        #     save_name
        # )

        nb_unique_clusters = df["cluster_label"].nunique()
        logging.info(f" Found {nb_unique_clusters} clusters\n")

        return df

    def p_cluster_faces_AHC(
        self,
        df: pd.DataFrame,
        model_name: str,
        affinity="cosine",
        linkage="complete",
        threshold=None,
        clusterColName=None,
    ) -> pd.DataFrame:
        from sklearn.cluster import AgglomerativeClustering

        if threshold is None:
            threshold = dst.findThreshold(model_name=model_name, distance_metric=affinity)

        logging.info("algorithm: AHC")
        logging.info(f"affinity: {affinity}")
        logging.info(f"linkage: {linkage}")
        logging.info(f"threshold: {threshold}")

        clustering = AgglomerativeClustering(
            distance_threshold=threshold, affinity=affinity, linkage=linkage, n_clusters=None
        )

        if not clusterColName:
            clusterColName = f"{model_name}_representation"

        t0 = perf_counter()
        X = np.array(df[clusterColName].to_list())
        res = clustering.fit(X)
        t1 = perf_counter()
        logging.info(f"Clustering faces took {t1 - t0} seconds")
        df["cluster_label"] = res.labels_

        return df

    def p_cluster_faces_DBSCAN(
        self, df: pd.DataFrame, model_name, min_samples=2, distance_metric="cosine", threshold=None, clusterColName=None
    ) -> pd.DataFrame:
        from sklearn.cluster import DBSCAN

        if threshold is None:
            threshold = dst.findThreshold(model_name=model_name, distance_metric=distance_metric)

        logging.info("algorithm: DBSCAN")
        logging.info(f"min_samples: {min_samples}")
        logging.info(f"distance_metric: {distance_metric}")
        logging.info(f"threshold: {threshold}")

        clustering = DBSCAN(eps=threshold, metric=distance_metric, min_samples=min_samples)

        if not clusterColName:
            clusterColName = f"{model_name}_representation"

        t0 = perf_counter()
        X = np.array(df[clusterColName].to_list())
        res = clustering.fit(X)
        t1 = perf_counter()
        logging.info(f"Clustering faces took {t1 - t0} seconds")
        df["cluster_label"] = res.labels_

        return df

    def run(self, model_name="Facenet512", clustering_algo="AHC", **kwargs):

        # rotate images
        self.p_rotate_images(self.src_folder)

        # extract faces
        df = self.p_extract_faces(self.src_folder, self.dst_folder, self.df_folder, threshold=0.95)

        # read df_confidences
        # df = pd.read_csv(self.df_folder / "confidences.csv")

        # only use first half of the images
        # df = df.iloc[: int(df.shape[0] / 2)]

        # use second half of the images
        # df = df.iloc[int(df.shape[0] / 2) :]

        # load the images
        df["img_bgr"] = df["image"].apply(lambda x: cv2.imread(f"{str(self.dst_folder)}/{x}"))

        # filter faces
        df = self.p_filter_faces(df, self.df_folder)
        # df = pd.read_csv(df_folder / "keep.csv")

        # representation of the faces
        df = df[df["keep"] == True]
        # df["img_bgr"] = df["image"].apply(lambda x: cv2.imread(f"{str(dst_folder)}/{x}"))

        df = self.p_represent_faces(df, self.df_folder, model_name=model_name)
        # df = pd.read_csv(
        #     self.df_folder / f"keep_representation_{model_name}.csv",
        #     index_col=0,
        #     converters={f"{model_name}_representation": json.loads},
        # )

        # cluster faces
        # df = self.p_cluster_faces(df, self.df_folder, model_name=model_name, clustering_algo=clustering_algo, **kwargs)


def main():

    base_path = pathlib.Path("/media/bao/t7/la_lib_dataset")
    # base_path = pathlib.Path("dataset")
    src_folder = base_path / "img3"
    save_folder = base_path / "120k-160k"
    save_folder.mkdir(exist_ok=True, parents=True)

    dst_folder = base_path / "faces3"
    dst_folder.mkdir(exist_ok=True, parents=True)
    df_folder = save_folder / "df"
    df_folder.mkdir(exist_ok=True, parents=True)
    log_folder = save_folder / "log"
    log_folder.mkdir(exist_ok=True, parents=True)

    faceClusteringPipeline = FaceClusteringPipeline(src_folder, dst_folder, df_folder, log_folder)
    faceClusteringPipeline.run(
        model_name="Facenet512", clustering_algo="DBSCAN", distance_metric="cosine", min_samples=5, threshold=0.24
    )


if __name__ == "__main__":
    # /home/bao/miniconda3/envs/snowflakes/bin/python /home/bao/Projects/master-thesis/liberte_archive/face_clustering_pipeline.py
    main()
