import pathlib
import cv2
from PIL import Image
from tqdm import tqdm

# https://www.dpreview.com/forums/post/25411567
rotation_degree = {6: cv2.ROTATE_90_CLOCKWISE, 3: cv2.ROTATE_180, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_images(folder_path: str) -> int:
    """
    rotate images in folder_path according to the EXIF orientation

    @param folder_path: str path to the folder containing images
    @return: int number of images rotated
    """

    src_folder = pathlib.Path(folder_path)

    img_list = list(src_folder.glob("*"))
    processed = 0

    for file in tqdm(img_list):

        if file.suffix not in [".jpg", ".png"]:
            continue

        try:
            exif = Image.open(file).getexif()
        except:
            exif = None

        if exif is None:
            continue

        if 274 in exif:
            orientation = exif[274]
        else:
            orientation = 0

        if orientation not in [3, 6, 8]:
            continue

        img = cv2.imread(str(file))
        img = cv2.rotate(img, rotation_degree[orientation])
        cv2.imwrite(str(file), img)
        processed += 1

    return processed
