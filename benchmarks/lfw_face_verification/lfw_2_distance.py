from deepface.basemodels import VGGFace, Facenet, OpenFace, FbDeepFace, ArcFace, Facenet512, DeepID, DlibWrapper
from deepface.basemodels import SFaceWrapper
from deepface.commons import distance as dst

from tqdm.autonotebook import tqdm
import pandas as pd
import json


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

distance_metrics = [
    "cosine",
    "euclidean",
    "euclidean_l2",
]

for model_name in tqdm(models.keys()):
    csv_path = f"results/intermediary/{model_name}_representations.csv"
    out_path = f"results/distances/{model_name}_distances.csv"

    img1_colname = f"{model_name}_img1"
    img2_colname = f"{model_name}_img2"

    df = pd.read_csv(
        f"results/intermediary/{model_name}_representations.csv",
        index_col=0,
        converters={img1_colname: json.loads, img2_colname: json.loads},
    )

    # calculate the distance metrics
    df["dst_cosine"] = df.apply(lambda x: dst.findCosineDistance(x[img1_colname], x[img2_colname]), axis=1)
    df["dst_euclidean"] = df.apply(lambda x: dst.findEuclideanDistance(x[img1_colname], x[img2_colname]), axis=1)
    df["dst_euclidean_l2"] = df.apply(
        lambda x: dst.findEuclideanDistance(dst.l2_normalize(x[img1_colname]), dst.l2_normalize(x[img2_colname])),
        axis=1,
    )

    # compare distance metrics with threshold
    for distance_metric in distance_metrics:
        threshold = dst.findThreshold(model_name=model_name, distance_metric=distance_metric)
        df[f"y_pred_{distance_metric}"] = df[f"dst_{distance_metric}"] <= threshold

    df.to_csv(out_path)
