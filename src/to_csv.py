import glob
import json
import csv

data = []
count = 1
reindexed_id = 1
flame_id = 1
length = 63
files = []


# jsonデータをdataに保存
def read_json(i):
    global flame_id, files
    for file in files:
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        if not json_load['people']:
            data.append([flame_id, reindexed_id, 0, 0, 0])
        else:
            json_data = json_load['people'][0]['hand_right_keypoints_2d']
            data.append([flame_id, reindexed_id, json_data[i], json_data[i + 1], json_data[i + 2]])
        flame_id += 1


# データをcsvに書き込む
def write_csv():
    file_name = "../csv/json_data" + str(count) + ".csv"
    with open(file_name, "w", newline="") as f:
        csv.writer(f).writerows(data)


# jsonフォルダのファイルを取得
def to_csv():
    global reindexed_id, files
    data.append(['flame_id', 'reindexed_id', 'x', 'y', 'credibility'])
    for img_number in range(5496, 5505, 1):
        files = glob.glob("../json/iMG_" + str(img_number) + "*")
        read_json(i)
        reindexed_id += 1
    write_csv()


for i in range(0, length, 3):
    to_csv()
    data = []
    count += 1
    flame_id = 1
    reindexed_id = 0
