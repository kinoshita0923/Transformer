import glob
import json
import csv

# 右手のデータの要素数を取得
length = 0
data = []
files = []
path = ""
count = 1

# jsonデータをdataに保存
def json_read(i):
    for file in files:
        json_open = open(file, 'r')
        json_load = json.load(json_open)
        if json_load['people'] == []:
            continue
        json_data = json_load['people'][0]['hand_right_keypoints_2d']
        data.append([json_data[i], json_data[i + 1], json_data[i + 2]])

# データをcsvに書き込む
def write_csv():
    file_name = "./csv/" + path + "/json_data" + str(count) + ".csv"
    with open(file_name, "w", newline = "") as f:
        csv.writer(f).writerows(data)

# 要素数の個数と同じ数のcsvファイルを作成
def create_csv():
    global count
    for i in range(0, length, 3):
        json_read(i)
        write_csv()
        data = []
        count+=1
    count = 1

# jsonフォルダのファイルを取得
for i in range(5496, 5505, 1):
    print(i)
    length = len(json.load(open("json\IMG_" + str(i) + "_000000000000_keypoints.json"))['people'][0]['hand_right_keypoints_2d'])
    files = glob.glob("./json/IMG_" + str(i) + "*")
    path = "IMG_" + str(i)
    create_csv()
    files = []