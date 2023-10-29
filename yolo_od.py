from ultralytics import YOLO

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

model = YOLO("yolov8x.pt")

def YOLO_detections(frame):
    detections = model(frame)[0]
    detecns = []
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        # confidence = data[4]
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        if class_id == 1:
            class_name = 'person'
            detecns.append([class_name, [xmin, ymin, xmax, ymax]])
        elif class_id == 3:
            class_name = 'car'
            detecns.append([class_name, [xmin, ymin, xmax, ymax]])
        elif class_id == 4:
            class_name = 'motorcycle'
            detecns.append([class_name, [xmin, ymin, xmax, ymax]])
        elif class_id == 6:
            class_name == 'bus'
            detecns.append([class_name, [xmin, ymin, xmax, ymax]])
        elif class_id == 8:
            class_name = 'truck'
            detecns.append([class_name, [xmin, ymin, xmax, ymax]])
        
        

        

    return detecns

def parse(promp):
    input_list = promp
    output_list = []

    for it in input_list:
        for item in it:
            # Check if the item contains ':' character
            if ':' in item:
                # Split the item by ':' and take the first part as the key
                key = item.split(':')[0].strip()

                # Check if the item contains '[' character, indicating a list
                if '[' in item:
                    try:
                        # Extract the value by evaluating the part after ':'
                        value = eval(item.split(':')[-1].strip())
                        # Append the key-value pair as a list to the output list
                        output_list.append([key, value])
                    except (ValueError, SyntaxError):
                          print(f"Skipping invalid item: {item}")
            else:
                return
        #   print(f"Skipping item without ':': {item}")
    return output_list
# print(output_list)

