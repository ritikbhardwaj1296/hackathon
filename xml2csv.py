import os
import shutil
import xml.etree.ElementTree as ET
import json
import csv  # Import the csv module

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

IMAGE_HEIGHT=964
IMAGE_WIDTH=1280

def grids(bbox_coordinates):
    b_center = (int((bbox_coordinates[2] - bbox_coordinates[0]) / 2), int((bbox_coordinates[3] - bbox_coordinates[1]) / 2))
    quad=""
    if b_center[1] <= IMAGE_HEIGHT/2:
        if b_center[0] <= IMAGE_WIDTH/2:
            if b_center[0] <= IMAGE_WIDTH/4:
                quad="Top_leftmost"
            else:
                quad="Top_center_left"
        else:
            if b_center[0] > IMAGE_WIDTH/2 + IMAGE_WIDTH/4:
                quad="Top_rightmost"
            else:
                quad="Top_center_right"
    else:
        if b_center[0] <= IMAGE_WIDTH/2:
            if b_center[0] <= IMAGE_WIDTH/4:
                quad="Bottom_leftmost"
            else:
                quad="Bottom_center_left"
        else:
            if b_center[0] > IMAGE_WIDTH/2 + IMAGE_WIDTH/4:
                quad="Bottom_rightmost"
            else:
                quad="Bottom_center_right"

    return quad




# Function to get captions from the VQA model
def get_caption(img_path, text):
    image = Image.open(img_path)
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]

# Function to extract object names and bounding box coordinates from an XML file
def extract_objects_with_coordinates(xml_file, image_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects_with_coordinates = []

    objects_with_coordinates.append(f"weather:{get_caption(image_file, 'What is the weather?')}")
    objects_with_coordinates.append(f"Time : {get_caption(image_file, 'What time of the day is it?')}")
    # objects_with_coordinates.append(f"Crowd : {get_caption(image_file, 'Is there crowd in this image?')}.")
    #objects_with_coordinates.append(f"Tress: {get_caption(image_file, 'Are there trees in scene?')}.")


    # objects_with_coordinates.append(f"{get_caption(image_file, 'Are there any trees?')} trees"

    for obj in root.findall(".//object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox_coordinates = (
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)
        )
        # area = abs(bbox_coordinates[2] - bbox_coordinates[0]) * abs(bbox_coordinates[3] - bbox_coordinates[1])
        # centre = (int((bbox_coordinates[2] - bbox_coordinates[0]) / 2), int((bbox_coordinates[3] - bbox_coordinates[1]) / 2), area)
        # quadrant = grids(bbox_coordinates)
        object_string = f"{name}: {[bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2], bbox_coordinates[3]]}"
        objects_with_coordinates.append(object_string)

    # if not objects_with_coordinates:
    #     objects_with_coordinates.append("No object at bounding box centre coordinate and area is (0, 0, 0)")

 
    return objects_with_coordinates

# Function to convert XML annotations to CSV format
def convert_to_csv(main_directory, output_csv):
    # Create a CSV file and write the header
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["file_name", "text"])

        merged_images_folder = os.path.join(main_directory, "merged_images")
        if not os.path.exists(merged_images_folder):
            os.makedirs(merged_images_folder)

        xml_folder = os.path.join(main_directory, "xml_folders")
        image_folder = os.path.join(main_directory, "image_folders")

        for parent_folder in os.listdir(xml_folder):
            parent_folder_path = os.path.join(xml_folder, parent_folder)
            cnt = 0
            if os.path.isdir(parent_folder_path):
                for filename in os.listdir(parent_folder_path):
                    if filename.endswith('.xml'):
                        xml_file = os.path.join(parent_folder_path, filename)
                        file_name = os.path.splitext(filename)[0]
                        combined_file_name = f"{parent_folder}-{file_name}"
                        image_subfolder = os.path.join(image_folder, parent_folder)
                        image_file = os.path.join(image_subfolder, f"{file_name}.jpg")
                        objects_with_coordinates = extract_objects_with_coordinates(xml_file, image_file)
                        if os.path.exists(image_file):
                            new_image_name = os.path.join(merged_images_folder, f"{combined_file_name}.jpg")
                            shutil.copy(image_file, new_image_name)
                        # Write data to CSV
                        csv_writer.writerow([combined_file_name + '.jpg', ",".join(objects_with_coordinates)])
                    print(cnt, end='\r')
                    cnt += 1

if __name__ == "__main__":
    main_directory = "/home/rbccps/Desktop/Projects/Hackathon/Datasets/IDD/idd-detection/main_data/"
    output_csv = os.path.join(main_directory, "metadata.csv")
    convert_to_csv(main_directory, output_csv)
