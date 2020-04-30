import os
import cv2
import shutil


src_folder = "test_img/origin"
kps_folder = "test_img/kps"
yes_folder = "test_img/selected"
unsure_folder = "test_img/unsure"
no_folder = "test_img/unselected"

delect_all = input("Delete all the previous record? Enter 'y' to delete")
if delect_all == "y":
    make_sure = input("Are you sure to delete? Enter 'y' to make sure")
    if make_sure == "y":
        try:
            os.remove(no_folder)
            os.remove(yes_folder)
            os.remove(unsure_folder)
            os.remove("processed.txt")
        except:
            pass


os.makedirs(yes_folder,exist_ok=True)
os.makedirs(unsure_folder,exist_ok=True)
os.makedirs(no_folder, exist_ok=True)
processed_file = []
if os.path.exists("processed.txt"):
    with open("processed.txt") as f:
        processed_file = [line[:-1] for line in f.readlines()]


past_img = ""

file = open("processed.txt", "a+")
for img_name in os.listdir(src_folder):
    if img_name in processed_file:
        continue
    img = cv2.imread(os.path.join(kps_folder, img_name))
    img = cv2.resize(img, (540, 360))
    cv2.imshow("img", img)
    cv2.moveWindow("img", 1000, 100)
    cv2.waitKey(1)
    while True:
        choose = input("Choose this image? ")
        if choose == "y":
            shutil.copy(os.path.join(src_folder, img_name), os.path.join(yes_folder, img_name))
            with open("processed.txt", "a+") as file:
                file.write(img_name)
                file.write("\n")
            past_img = img_name
            break
        elif choose == "n":
            shutil.copy(os.path.join(src_folder, img_name), os.path.join(no_folder, img_name))
            with open("processed.txt", "a+") as file:
                file.write(img_name)
                file.write("\n")
            past_img = img_name
            break
        elif choose == "u":
            shutil.copy(os.path.join(src_folder, img_name), os.path.join(unsure_folder, img_name))
            with open("processed.txt", "a+") as file:
                file.write(img_name)
                file.write("\n")
            past_img = img_name
            break
        elif choose == "p":
            with open("suspect.txt", "a+") as suspect_file:
                suspect_file.write(past_img)
                suspect_file.write("\n")
            print("The previous image has been recorded")
        else:
            print("The enter is wrong")
            continue

