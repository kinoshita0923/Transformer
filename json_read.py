import glob
import json
import csv

# 右手のデータの要素数を取得
length = 0
data = []
files = []
path = ""

# jsonデータをdataに保存
def json_read(i):
    for index, file in enumerate(files):
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        data.append([index, json_load['people'][0]['hand_right_keypoints_2d'][i]])

# データをcsvに書き込む
def write_csv(i):
    file_name = "./csv/" + path + "/json_data" + str(i) + ".csv"
    with open(file_name, "w", newline = "") as f:
        csv.writer(f).writerows(data)

# 要素数の個数と同じ数のcsvファイルを作成
def create_csv():
    for i in range(length):
        json_read(i)
        write_csv(i+1)
        data = []

# jsonフォルダのファイルを取得
for i in range(5496, 5505, 1):
    length = len(json.load(open("json\IMG_" + str(i) + "_000000000000_keypoints.json"))['people'][0]['hand_right_keypoints_2d'])
    files = glob.glob("./json/IMG_" + str(i) + "*")
    path = "IMG_" + str(i)
    create_csv()
    files = []