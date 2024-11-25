# -*- coding = utf-8 -*-
# @Time : 2024/11/4 14:39
# @Author : 王砚轩
# @File : trajMatchMergingType.py
# @Software: PyCharm
import os
from loguru import logger
import pandas as pd



def main():
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    preProcessFile = pd.read_csv(assetPath+"compressed_data.csv")
    mergingTypeFile = pd.read_csv(assetPath+"mergingDataNew200m.csv")

    # 将mergingtype数据储存为二维set
    dictMergingType = dict()
    for file in mergingTypeFile.itertuples():
        dictMergingType[(file.recordingId,file.trackId)] = file.MergingType
    logger.info(dictMergingType)


    # 将mergingtype与压缩后轨迹匹配
    result = []
    for file in preProcessFile.itertuples():
        if (file.recordingId, file.trackId) in dictMergingType:
            result.append(dictMergingType[(file.recordingId, file.trackId)])
        else:
            result.append(None)

    preProcessFile['MergingType'] = result
    print(preProcessFile.columns[0])
    preProcessFile.drop(columns=['Unnamed: 0'],inplace=True)
    print(preProcessFile.columns[0])
    preProcessFile.to_csv(assetPath+"trajMatchMergingType.csv")




if __name__ == '__main__':
    main()