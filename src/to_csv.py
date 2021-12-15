import glob
import json
import csv

data = []
count = 1


# jsonデータをdataに保存
def read_json(i):
    for file in files:
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        if not json_load['people']:
            data.append([0, 0, 0])
        else:
            json_data = json_load['people'][0]['hand_right_keypoints_2d']
            data.append([json_data[i], json_data[i + 1], json_data[i + 2]])


# データをcsvに書き込む
def write_csv():
    file_name = "../csv/" + path + "/json_data" + str(count) + ".csv"
    with open(file_name, "w", newline="") as f:
        csv.writer(f).writerows(data)


# 要素数の個数と同じ数のcsvファイルを作成
def to_csv():
    global count, files, data
    for i in range(0, length, 3):
        read_json(i)
        write_csv()
        data = []
        count += 1
    count = 1
    files = []


# jsonフォルダのファイルを取得
for img_number in range(5496, 5505, 1):
    json_file = json.load(open("../json/IMG_" + str(img_number) + "_000000000000_keypoints.json"))
    file_number = 1
    while not json_file['people']:
        json_file = json.load(open("../json/IMG_" + str(img_number) + "_{0:12d}".format(file_number)
                              + str(file_number) + "_keypoints.json"))
        file_number += 1
    length = len(json_file['people'][0]['hand_right_keypoints_2d'])
    files = glob.glob("../json/IMG_" + str(img_number) + "*")
    path = "IMG_" + str(img_number)
    to_csv()
