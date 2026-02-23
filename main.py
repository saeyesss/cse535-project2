import os
import cv2
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from handshape_feature_extractor import HandShapeFeatureExtractor

class GestureDetail:
    def __init__(self, output_label, train_keys, test_keys):
        self.output_label = output_label
        self.train_keys = train_keys
        self.test_keys = test_keys

class GestureFeature:
    def __init__(self, gesture_detail, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature

gesture_data = [
    GestureDetail("0",  ["Num0"],      ["0"]),
    GestureDetail("1",  ["Num1"],      ["1"]),
    GestureDetail("2",  ["Num2"],      ["2"]),
    GestureDetail("3",  ["Num3"],      ["3"]),
    GestureDetail("4",  ["Num4"],      ["4"]),
    GestureDetail("5",  ["Num5"],      ["5"]),
    GestureDetail("6",  ["Num6"],      ["6"]),
    GestureDetail("7",  ["Num7"],      ["7"]),
    GestureDetail("8",  ["Num8"],      ["8"]),
    GestureDetail("9",  ["Num9"],      ["9"]),
    GestureDetail("10", ["FanDown"],   ["DecreaseFanSpeed", "DecereaseFanSpeed"]),
    GestureDetail("11", ["FanOff"],    ["FanOff"]),
    GestureDetail("12", ["FanOn"],     ["FanOn"]),
    GestureDetail("13", ["FanUp"],     ["IncreaseFanSpeed"]),
    GestureDetail("14", ["LightOff"],  ["LightOff"]),
    GestureDetail("15", ["LightOn"],   ["LightOn"]),
    GestureDetail("16", ["SetThermo"], ["SetThermo"]),
]



def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = max(0, total_frames // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def extract_feature(video_path, extractor):
    frame = extract_middle_frame(video_path)
    if frame is None:
        return None
    feature = extractor.extract_feature(frame)
    if feature is None:
        return None
    return feature.flatten()





def decide_gesture_by_train_filename(filename):
    prefix = filename.split('_')[0]
    for gd in gesture_data:
        if prefix in gd.train_keys:
            return gd
    return None


def decide_gesture_by_test_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('-', 2)
    if len(parts) < 3:
        return None
    suffix = parts[2]
    for gd in gesture_data:
        if suffix in gd.test_keys:
            return gd
    return None





def main():

    extractor = HandShapeFeatureExtractor.get_instance()
    train_path = "traindata/"
    test_path = "test/"

    featureVectorList = []
    for file in sorted(os.listdir(train_path)):
        if not file.lower().endswith(".mp4"):
            continue

        gesture_detail = decide_gesture_by_train_filename(file)
        if gesture_detail is None:
            continue

        video_path = os.path.join(train_path, file)
        feature = extract_feature(video_path, extractor)

        if feature is not None:
            featureVectorList.append(
                GestureFeature(gesture_detail, feature)
            )

    
    results = []

    for test_file in sorted(os.listdir(test_path)):
        if not test_file.lower().endswith(".mp4"):
            continue

        test_video_path = os.path.join(test_path, test_file)
        test_feature = extract_feature(test_video_path, extractor)

        if test_feature is None:
            results.append(0)
            continue

        max_similarity = -1
        recognized_label = 0

        for fv in featureVectorList:
            similarity = cosine_similarity(
                test_feature.reshape(1, -1),
                fv.extracted_feature.reshape(1, -1)
            )[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                recognized_label = int(fv.gesture_detail.output_label)

        results.append(recognized_label)

    
    with open("Results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for label in results:
            writer.writerow([label])


if __name__ == "__main__":
    main()