from typing import List, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
from requests import Session
from io import BytesIO
from PIL import Image, ImageEnhance
from absl import logging
import hashlib
import time
import datetime
import sys
from collections import defaultdict
import numpy as np

GRAND_CANYON_SOUTH_ENTRANCE_WEBCAM = "https://www.nps.gov/webcams-grca/camera.jpg"
PRETRAINED_MODEL = "yolo11n.pt"
SLEEP = 300
ALLOWED_OBJECTS = ["bus", "car", "truck"]

def get_webcam_image(session: Session) -> BytesIO:
    response = session.get(GRAND_CANYON_SOUTH_ENTRANCE_WEBCAM)
    return BytesIO(response.content)

def get_num_filtered_detections(model: YOLO, image: Image, save_to_file: str = "") -> int:
    result = model([image])[0]
    detected_objects = result.boxes.cls
    name_mapping = result.names

    readable_detected_objects = list(map(lambda object: name_mapping[object.item()], detected_objects))
    logging.info("Objects detected: {}".format(readable_detected_objects))

    filtered_objects = list(filter(lambda object: object in ALLOWED_OBJECTS, readable_detected_objects))
    logging.info("Number of vehicles detected: {}".format(len(filtered_objects)))

    if save_to_file: result.save(filename=save_to_file)
    return len(filtered_objects)

def round_float(num_to_round):
    return np.round([num_to_round])[0]

def find_optimal_settings(model: YOLO, image: Image, save_to_file: str = "") -> List[Tuple]:
    settings_map = defaultdict(list)
    num_detections_original = get_num_filtered_detections(model=model, image=image, save_to_file=save_to_file)
    
    for brightness in np.arange(.2, 2.0, .1, dtype=float):
        enhancer = ImageEnhance.Brightness(image)
        darkened_image = enhancer.enhance(brightness)

        for contrast in np.arange(.2, 2.0, .1, dtype=float):
            enhancer = ImageEnhance.Contrast(darkened_image)
            contrasted_image = enhancer.enhance(contrast)

            for sharp in np.arange(.2, 2.0, .1, dtype=float):
                enhancer = ImageEnhance.Sharpness(contrasted_image)
                sharpened_image = enhancer.enhance(sharp)

                num_detections_new = get_num_filtered_detections(model=model, image=sharpened_image, save_to_file=save_to_file+"sharp")

                if num_detections_new > num_detections_original: settings_map[num_detections_new].append((round_float(brightness), round_float(contrast), round_float(sharp)))

    logging.info(settings_map)

    return settings_map(max(settings_map))

def main():
    logging.set_verbosity(logging.INFO)
    model = YOLO(PRETRAINED_MODEL)
    logging.info(f"Initialized pretrained model {PRETRAINED_MODEL}.")

    match len(sys.argv):
        case 1:
            logging.info("Using online image.")
            previous_image_hash = ""
            session = Session()
            while True:
                logging.info("Starting loop.")
                process_time = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
                image_bytesio = get_webcam_image(session=session)
                image_hash = hashlib.md5(image_bytesio.getvalue()).hexdigest()

                if image_hash != previous_image_hash:
                    logging.info("New image detected.")
                    image = Image.open(image_bytesio)
                    get_num_filtered_detections(model=model, image=image, save_to_file=process_time+".jpg")

                    filename = f"original-{process_time}.jpg"
                    image.save(filename)
                    logging.info(f"Wrote image with filename {filename}.")

                    previous_image_hash = image_hash
                else:
                    logging.info("Image has not changed.")

                logging.info("Sleeping for {}.".format(SLEEP))
                time.sleep(SLEEP)
        case 2:
            logging.info("Using supplied image path.")
            process_time = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
            image = Image.open(sys.argv[1])
            find_optimal_settings(model=model, image=image, save_to_file=process_time)
        case 4:
            logging.info("Using online image with supplied settings.")
            session = Session()
            process_time = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
            image_bytesio = get_webcam_image(session=session)
            image = Image.open(image_bytesio)
            original_filename = f"result-{process_time}-original.jpg"
            image.save(original_filename)
            get_num_filtered_detections(model=model, image=image, save_to_file=process_time+".jpg")
            enhancer = ImageEnhance.Brightness(image)
            darkened_image = enhancer.enhance(float(sys.argv[1]))
            enhancer = ImageEnhance.Contrast(darkened_image)
            contrasted_image = enhancer.enhance(float(sys.argv[2]))
            enhancer = ImageEnhance.Sharpness(contrasted_image)
            sharpened_image = enhancer.enhance(float(sys.argv[3]))
            enhanced_filename = f"result-{process_time}-enhanced.jpg"
            sharpened_image.save(enhanced_filename)
            get_num_filtered_detections(model=model, image=sharpened_image, save_to_file=process_time+"enhance.jpg")
        case 5:
            logging.info("Using supplied image path and settings.")
            process_time = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp()))
            image = Image.open(sys.argv[1])
            enhancer = ImageEnhance.Brightness(image)
            darkened_image = enhancer.enhance(float(sys.argv[2]))
            enhancer = ImageEnhance.Contrast(darkened_image)
            contrasted_image = enhancer.enhance(float(sys.argv[3]))
            enhancer = ImageEnhance.Sharpness(contrasted_image)
            sharpened_image = enhancer.enhance(float(sys.argv[4]))
            get_num_filtered_detections(model=model, image=sharpened_image, save_to_file=process_time)
        case _:
            logging.info("What are you doing??")

if __name__ == "__main__":
    main()
