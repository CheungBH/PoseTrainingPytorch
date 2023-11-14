import os
import json
from PIL import Image

json_file = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/result.json"
image_folder = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/images"
output_folder = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d"
scale_factor = 0.2


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

                scale_factor /= 2
                width_scale = bbox[2] * scale_factor
                height_scale = bbox[3] * scale_factor

                left = max(0, bbox[0] - width_scale)
                top = max(0, bbox[1] - height_scale)
                right = min(image.width, bbox[0] + bbox[2] + width_scale)
                bottom = min(image.height, bbox[1] + bbox[3] + height_scale)


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
