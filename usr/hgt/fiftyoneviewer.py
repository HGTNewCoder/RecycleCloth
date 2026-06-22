
import csv
import json
import os

import fiftyone as fo
import tqdm

train_image_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/deepfashion2_original_images/train/image/'
validation_image_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/deepfashion2_original_images/validation/image/'
train_annos_image_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/deepfashion2_original_images/train/annos/'
validation_annos_image_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/deepfashion2_original_images/validation/annos/'
train_csv_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/img_info_dataframes/train.csv'
validation_csv_path = f'/home/ubuntu/RecycleCloth/Datasets/DeepFashion2/img_info_dataframes/validation.csv'


def load_image_metadata(csv_path):
    metadata_by_stem = {}

    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            stem = os.path.splitext(os.path.basename(row['path']))[0]
            if stem not in metadata_by_stem:
                metadata_by_stem[stem] = {
                    'width': int(row['img_width']),
                    'height': int(row['img_height']),
                }

    return metadata_by_stem


def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox

    x = max(0.0, x_min / width)
    y = max(0.0, y_min / height)
    w = max(0.0, (x_max - x_min) / width)
    h = max(0.0, (y_max - y_min) / height)

    return [x, y, w, h]


image_metadata_by_stem = load_image_metadata(validation_csv_path)

dataset_name = "DeepFashion2-validation"

if fo.dataset_exists(dataset_name):
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = fo.Dataset.from_dir(
        dataset_dir=validation_image_path,
        dataset_type=fo.types.ImageDirectory,
        name=dataset_name,
        persistent=True,
    )
print(dataset)

for sample in tqdm.tqdm(dataset):
    sample_root = os.path.splitext(os.path.basename(sample.filepath))[0]
    sample_ann_path = os.path.join(validation_annos_image_path, sample_root + '.json')
    detections = []
    polylines = []

    image_metadata = image_metadata_by_stem.get(sample_root)

    if image_metadata is None:
        continue

    if not os.path.exists(sample_ann_path):
        continue

    with open(sample_ann_path, 'r') as file:
        data = json.load(file)

    for item_name, item in data.items():
        if not item_name.startswith('item'):
            continue

        item_index = int(item_name.replace('item', ''))

        if 'bounding_box' not in item:
            continue  # skip items missing bounding box

        segmentation_list = item.get('segmentation', [])
        if not segmentation_list:
            continue
        segmentation = segmentation_list[0]

        if len(segmentation) % 2 != 0:
            continue  # malformed segmentation, skip

        bbox_adjusted = normalize_bbox(
            item['bounding_box'],
            image_metadata['width'],
            image_metadata['height'],
        )

        poly_points = []

        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / image_metadata['width']
            y = segmentation[i + 1] / image_metadata['height']
            poly_points.append([x, y])

        detections.append(
            fo.Detection(
                bounding_box=bbox_adjusted,
                label=item['category_name'],
            )
        )

        polylines.append(
            fo.Polyline(
                label=item['category_name'],
                item_index=item_index,
                points=[poly_points],
                closed=True,
                filled=True
            )
        )

    sample['ground_truth'] = fo.Detections(detections=detections)
    sample['segmentations'] = fo.Polylines(polylines=polylines)
    sample.save()

dataset.app_config.color_scheme = fo.ColorScheme(
    fields=[
        {
            "path": "segmentations",
            "colorByAttribute": "item_index",
            "valueColors": [
                {"value": "1", "color": "#39A845"},
                {"value": "2", "color": "#FF0000"},
            ],
        }
    ]
)
dataset.save()

session = fo.launch_app(dataset, port = 5151)
session.wait()