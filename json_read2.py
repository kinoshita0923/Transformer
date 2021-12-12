import glob
import json
import csv

# 右手のデータの要素数を取得
length = 0
data = []
files = []
path = ""

# jsonデータをdataに保存
def json_read():
    for index, file in enumerate(files):
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        length = len(json_load['people'][0]['hand_right_keypoints_2d'])
        data.append([])
        for i in range(length):
            print(index,i)
            data[index].append(json_load['people'][0]['hand_right_keypoints_2d'][i])
            print(index, i)

# データをcsvに書き込む
def write_csv():
    file_name = "./csv2/" + path + ".csv"
    with open(file_name, "w", newline = "") as f:
        csv.writer(f).writerows(data)

# jsonフォルダのファイルを取得
for i in range(5496, 5505, 1):
    files = glob.glob("./json/IMG_" + str(i) + "*")
    path = "IMG_" + str(i)
    json_read()
    write_csv()