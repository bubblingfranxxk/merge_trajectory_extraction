# -*- coding = utf-8 -*-
# @Time : 2024/11/6 20:34
# @Author : 王砚轩
# @File : singleTrajLength.py
# @Software: PyCharm
import os
import pandas as pd
from config import laneletID

recordingMapToLocation = {
    "0": list(range(0, 19)),
    "1": list(range(19, 39)),
    "2": list(range(39, 53)),
    "3": list(range(53, 61)),
    "4": list(range(61, 73)),
    "5": list(range(73, 78)),
    "6": list(range(78, 93)),
}

endpoint = {
    "2": laneletID.lanlet2data["2"]["area2"]+laneletID.lanlet2data["2"]["area5"],
    "3": laneletID.lanlet2data["3"]["area2"]+laneletID.lanlet2data["3"]["area5"],
    "5": laneletID.lanlet2data["5"]["area2"]+laneletID.lanlet2data["5"]["area5"],
    "6": laneletID.lanlet2data["6"]["area2"]+laneletID.lanlet2data["6"]["area5"]
}
startpoint = {
    "2": laneletID.lanlet2data["2"]["area2"] + laneletID.lanlet2data["2"]["area5"],
    "3": laneletID.lanlet2data["3"]["area2"] + laneletID.lanlet2data["3"]["area5"],
    "5": laneletID.lanlet2data["5"]["area2"] + laneletID.lanlet2data["5"]["area5"],
    "6": laneletID.lanlet2data["6"]["area2"] + laneletID.lanlet2data["6"]["area5"]
}

def count_time_lengths(folder_path, output_file='time_lengths.csv'):
    time_lengths = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            recordingId = df['recordingId'][0]

            # 确定location
            location = "2" if recordingId in recordingMapToLocation["2"] \
                else "3" if recordingId in recordingMapToLocation["3"]\
                else "5" if recordingId in recordingMapToLocation["5"]\
                else "6"
            file_endpoint = endpoint[location]
            if df['laneletId'][len(df)-1] not in file_endpoint:
                continue
            time_lengths[file_name] = len(df)

    # Convert the dictionary to a DataFrame and save it to a CSV file
    time_lengths_df = pd.DataFrame(list(time_lengths.items()), columns=['File Name', 'Time Length'])
    time_lengths_df.to_csv(output_file, index=False)

    print(f'Time lengths saved to {output_file}')


# Example usage
folder_path = os.path.abspath('../../') + r'\asset\single_traj'
count_time_lengths(folder_path, os.path.abspath('../../')+r'\asset\time_lengths.csv')
