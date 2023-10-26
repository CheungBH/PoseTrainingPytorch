import os
import json
from PIL import Image

json_file = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/result.json"
image_folder = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/images"
output_folder = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/fall"
scale_num = 15


def scale(img_h, img_w, bbox, sf):
    assert len(sf) == 4, "You should assign 4 different factors value"
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = x_max - x_min, y_max - y_min
    # imgheight = img.shape[0]
    # imgwidth = img.shape[1]
    x_enlarged_min = max(0, x_min - width * sf[0] / 2)
    y_enlarged_min = max(0, y_min - height * sf[1] / 2)
    x_enlarged_max = min(img_w - 1, x_max + width * sf[2] / 2)
    y_enlarged_max = min(img_h - 1, y_max + height * sf[3] / 2)
    return [x_enlarged_min, y_enlarged_min, x_enlarged_max, y_enlarged_max]


with open(json_file, "r") as f:
    data = json.load(f)

image_files = os.listdir(image_folder)

os.makedirs(output_folder, exist_ok=True)

category_count = {}

for image_data in data["images"]:
    image_id = image_data["id"]
    file_name = image_data["file_name"]

    if file_name in image_files:
        image_path = os.path.join(image_folder, file_name)

        annotation = next((ann for ann in data["annotations"] if ann["image_id"] == image_id), None)
        if annotation:
            category_id = annotation["category_id"]
            bbox = annotation["bbox"]

            category_name = next((cat["name"] for cat in data["categories"] if cat["id"] == category_id), None)
            if category_name:

                category_folder = os.path.join(output_folder, category_name)
                os.makedirs(category_folder, exist_ok=True)

                if category_name in category_count:
                    category_count[category_name] += 1
                else:
                    category_count[category_name] = 1

                new_file_name = f"{os.path.splitext(file_name)[0]}_{category_name}_{category_count[category_name]}.jpg"

                image = Image.open(image_path)

                left = max(0, bbox[0] - scale_num)
                top = max(0, bbox[1] - scale_num)
                right = min(image.width, bbox[0] + bbox[2] + scale_num)
                bottom = min(image.height, bbox[1] + bbox[3] + scale_num)

                # left, top, right, bottom = scale(image.height, image.width, bbox, [0.2, 0.2, 0.2, 0.2])

                if left < right and top < bottom:
                    cropped_image = image.crop((left, top, right, bottom))

                    cropped_image.save(os.path.join(category_folder, new_file_name))
                else:
                    print(f"Invalid coordinates for cropping: {bbox}")
            else:
                print(f"Cannot find category name for category_id: {category_id}")
        else:
            print(f"No annotation found for image_id: {image_id}")
    else:
        print(f"Image file not found: {file_name}")
