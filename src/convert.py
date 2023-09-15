import os
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from pycocotools.coco import COCO
from supervisely.io.fs import get_file_name, get_file_size
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(desc=f"Downloading '{file_name_with_ext}' to buffer...", total=fsize) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    dataset_path = "dentalai.v2i.coco-segmentation"
    img_path = os.path.join(dataset_path, "imgs")
    batch_size = 50

    inst_test = os.path.join(dataset_path, "test", "_annotations.coco.json")
    inst_train = os.path.join(dataset_path, "train", "_annotations.coco.json")
    inst_valid = os.path.join(dataset_path, "valid", "_annotations.coco.json")

    instances = {"test": inst_test, "train": inst_train, "valid": inst_valid}

    label_names = ["Caries", "Cavity", "Crack", "Tooth"]

    def segm_fix(segm):
        geometry = []
        for i in range(0, len(segm) - 1, 2):
            cords = sly.PointLocation(segm[i + 1], segm[i])
            geometry.append(cords)
        return geometry

    def create_ann(image_path, img_dict):
        image_id = img_dict[image_path]
        labels = []
        img_height = images[image_id]["height"]
        img_wight = images[image_id]["width"]
        for label in annotations[image_id]:
            try:
                segm = label["segmentation"]
                segm_fixed = segm_fix(segm[0])
            except Exception:
                pass
            cat_id = label["category_id"]
            label_name = categories[cat_id]["name"]
            obj_class = meta.get_obj_class(label_name)
            geometry = sly.Polygon(segm_fixed)
            curr_label = sly.Label(geometry, obj_class)
            labels.append(curr_label)
        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_classes = [sly.ObjClass(name, sly.Polygon) for name in label_names]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=obj_classes)
    api.project.update_meta(project.id, meta.to_json())

    dataset_test = api.dataset.create(project.id, "test", change_name_if_conflict=True)
    dataset_train = api.dataset.create(project.id, "train", change_name_if_conflict=True)
    dataset_valid = api.dataset.create(project.id, "valid", change_name_if_conflict=True)

    def get_path(file, ds):
        return os.path.join(dataset_path, ds, file)

    for inst in instances:
        if inst == "test":
            dataset = dataset_test
        elif inst == "train":
            dataset = dataset_train
        else:
            dataset = dataset_valid
        instance_coco = COCO(instances[inst])
        categories = instance_coco.cats
        images = instance_coco.imgs
        indexes = list(images)
        annotations = instance_coco.imgToAnns
        progress = sly.Progress("Create dataset {}".format(dataset.name), len(indexes))
        for index_batch in sly.batched(indexes, batch_size=batch_size):
            img_paths = [get_path(images[i]["file_name"], inst) for i in index_batch]
            img_keys = [images[i]["id"] for i in index_batch]
            img_names_batch = [os.path.basename(img_path) for img_path in img_paths]
            img_dict = {img[0]: img[1] for img in zip(img_paths, img_keys)}
            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_paths)
            img_ids = [im_info.id for im_info in img_infos]
            anns_batch = [create_ann(image_path, img_dict) for image_path in img_paths]
            api.annotation.upload_anns(img_ids, anns_batch)
            progress.iters_done_report(len(img_names_batch))
    return project
