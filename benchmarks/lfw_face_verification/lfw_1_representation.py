from deepface import DeepFace
from deepface.basemodels import VGGFace, Facenet, OpenFace, FbDeepFace, ArcFace, Facenet512, DeepID, DlibWrapper
from deepface.basemodels import SFaceWrapper

from sklearn.datasets import fetch_lfw_pairs

from tqdm import tqdm

import pandas as pd
import numpy as np
import pathlib

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

lfw_pairs = fetch_lfw_pairs(subset="10_folds", color=True, resize=1)

X = lfw_pairs["pairs"]
y = lfw_pairs["target"]

# put X and y in a dataframe
df = pd.DataFrame({"pairs": list(X), "y_true": y})

# split the pairs into two columns
df["img1"] = df["pairs"].apply(lambda x: x[0])
df["img2"] = df["pairs"].apply(lambda x: x[1])
df = df.drop(columns=["pairs"], errors="ignore")

# convert to BGR
df["img1"] = df["img1"].apply(lambda x: x[..., ::-1])
df["img2"] = df["img2"].apply(lambda x: x[..., ::-1])

# split the dataframe into 10 folds
df["fold"] = np.repeat(np.arange(10), 600)

for model_name in tqdm(models.keys()):
    print(f"{model_name}")

    # create a folder for storing the results
    save_folder = f"results"
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    intermediary_folder = f"{save_folder}/intermediary"
    pathlib.Path(intermediary_folder).mkdir(parents=True, exist_ok=True)

    model = models[model_name]()

    df[f"{model_name}_img1"] = df["img1"].apply(
        lambda x: DeepFace.represent(
            img_path=x, model_name=model_name, model=model, enforce_detection=False, detector_backend="skip"
        )
    )
    df[f"{model_name}_img2"] = df["img2"].apply(
        lambda x: DeepFace.represent(
            img_path=x, model_name=model_name, model=model, enforce_detection=False, detector_backend="skip"
        )
    )

    df[[f"{model_name}_img1", f"{model_name}_img2", "y_true", "fold"]].to_csv(
        f"{intermediary_folder}/{model_name}_representations.csv"
    )

# df.to_csv(f"{save_folder}/representations.csv")
