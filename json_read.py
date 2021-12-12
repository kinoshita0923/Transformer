import glob
import json
import csv

files = glob.glob("./json/IMG_5496*")
data = []

def json_read():
    for file in files:
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        data.append([json_load['people'][0]['hand_right_keypoints_2d'][0], json_load['people'][0]['hand_right_keypoints_2d'][1]])

def writeCsv():
    with open("./json_data.csv", "w", newline = "") as f:
        csv.writer(f).writerows(data)

json_read()
writeCsv()