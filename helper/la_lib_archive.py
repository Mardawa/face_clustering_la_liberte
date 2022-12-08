from dataclasses import dataclass
import requests
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from tqdm.autonotebook import tqdm
import pathlib
from ast import literal_eval
from dataclasses import dataclass


API_USER = os.getenv("API_USER")
API_PASS = os.getenv("API_PASS")
URL = os.getenv("URL")


@dataclass
class LaLibAPI:

    df_folder: str

    def __post_init__(self):
        # create folder
        pathlib.Path(self.df_folder).mkdir(parents=True, exist_ok=True)

    def get_authToken(self):
        data = {"username": API_USER, "password": API_PASS}

        r = requests.post(URL + "/services/apilogin", data=data)
        return r.json()["authToken"]

    def get_headers(self):
        return {"Authorization": "Bearer " + self.get_authToken()}

    def search(self, start: int, num=50, save=False, save_name=None):

        params = {
            "q": "!description: * AND !otherConstraints:*",  # no descripiton and no otherConstraints
            "metadataToReturn": "description,otherConstraints,otherConditions",
            "facets": "assetDomain,kind",
            "facet.assetDomain.selection": "image",
            "num": num,
            "start": start,
        }

        headers = self.get_headers()
        s = requests.get(URL + "/services/search", params=params, headers=headers)
        resp = s.json()

        df = pd.DataFrame(resp["hits"])

        if save:
            if save_name is None:
                save_name = f"df_{start}_{num}.csv"
            df.to_csv(f"{df_folder}/{save_name}", index=False)

        return df

    # def set_metadata(self, id, metadata_field, metadata_value):
    #     headers = self.get_headers()

    #     params = {"q": f"id:{id}", metadata_field: metadata_value}

    #     r = requests.post(URL + "/services/updatebulk", params=params, headers=headers)
    #     return r

    def set_metadatas(self, ids, metadata_field, metadata_value):

        for i in tqdm(range(0, len(ids), 200)):
            headers = self.get_headers()
            # generate the query
            query = " OR ".join([f"id:{id}" for id in ids[i : i + 200]])

            params = {
                "q": query,
                metadata_field: metadata_value,
            }

            u = requests.post(URL + "/services/updatebulk", data=params, headers=headers)

    def retrieve_full_metedata(self, ids, n_per_query=250):
        headers = self.get_headers()

        df_resp = pd.DataFrame()

        # the query is split else the url is too long and the request fails
        for idx, i in tqdm(enumerate(range(0, len(ids), n_per_query)), total=len(ids) // n_per_query):

            # refresh the headers every 10 queries
            if idx % 10 == 0:
                headers = self.get_headers()

            # generate the query
            query = " OR ".join([f"id:{id}" for id in ids[i : i + n_per_query]])

            params = {
                "q": query,
                "metadataToReturn": "all",
                "facets": "assetDomain,kind",
                "facet.assetDomain.selection": "image",
                "num": n_per_query,
                "start": 0,
            }

            s = requests.get(URL + "/services/search", params=params, headers=headers)
            resp = s.json()
            df_tmp = pd.DataFrame(resp["hits"])
            df_resp = pd.concat([df_resp, df_tmp])

        return df_resp

    def mark_as_viewed(self, ids):
        self.set_metadatas(ids, "otherConstraints", "viewed")

    def download_from_df(self, df: pd.DataFrame, folder: str):
        headers = self.get_headers()
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            # refresh the token every 2000 images
            if index % 1000 == 0:
                headers = self.get_headers()

            downLoadUrl = row["originalUrl"]
            img_id = row["id"]
            assetType = row["metadata"].get("assetType")

            img = requests.get(downLoadUrl, headers=headers)

            with open(f"{folder}/{img_id}.{assetType}", "wb") as f:
                f.write(img.content)


if __name__ == "__main__":
    df_folder = "/media/bao/t7/la_lib_dataset/df"

    # retrieve the full metadata
    # api = LaLibAPI(df_folder=df_folder)
    # for i in tqdm(range(2, 17)):
    #     save_name = f"df{i}.csv"
    #     df = pd.read_csv(f"{df_folder}/{save_name}")
    #     ids = df["id"].tolist()
    #     df_full = api.retrieve_full_metedata(ids)
    #     df_full.to_csv(f"{df_folder}_w_metadata/df{i}.csv", index=False)

    # retrieve the ids and metadata
    # and mark them as viewed
    # for i in range(13, 17):
    #     save_name = f"df{i}.csv"
    #     api = LaLibAPI(df_folder=df_folder)
    #     df = api.search(start=0, num=10000, save=True, save_name=save_name)
    #     # df = pd.read_csv(f"{df_folder}/{save_name}")
    #     ids = df["id"].tolist()
    #     api.mark_as_viewed(ids)

    # csv_path = pathlib.Path("dataset/dataset_detail")
    # csv_files = csv_path.glob("*.csv")
    # for csv_file in csv_files:
    #     api = LaLibAPI(df_folder=df_folder)
    #     # df = api.search(start=0, num=10000, save=True, save_name=save_name)
    #     df = pd.read_csv(csv_file)
    #     ids = df["id"].tolist()
    #     api.mark_as_viewed(ids)

    # download the images
    # for i in tqdm(range(13, 17)):
    #     save_name = f"df{i}.csv"
    #     df = pd.read_csv(f"{df_folder}/{save_name}", converters={"metadata": literal_eval})
    #     api = LaLibAPI(df_folder=df_folder)
    #     api.download_from_df(df, folder=f"/media/bao/t7/la_lib_dataset/img3")
