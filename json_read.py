import glob
import json
import csv

# jsonフォルダのファイルを取得
files = glob.glob("./json/IMG_5496*")

# 右手のデータの要素数を取得
length = len(json.load(open(files[0]))['people'][0]['hand_right_keypoints_2d'])
data = []

# jsonデータをdataに保存
def json_read(i):
    for index, file in enumerate(files):
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        data.append([index, json_load['people'][0]['hand_right_keypoints_2d'][i]])

# データをcsvに書き込む
def write_csv(i):
    file_name = "./csv/json_data" + str(i) + ".csv"
    with open(file_name, "w", newline = "") as f:
        csv.writer(f).writerows(data)

# 要素数の個数と同じ数のcsvファイルを作成
for i in range(length):
    json_read(i)
    write_csv(i+1)
    data = []