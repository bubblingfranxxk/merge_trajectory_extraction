# -*- coding = utf-8 -*-
# @Time : 2023/12/26 20:30
# @Author : 王砚轩
# @File : laneletID.py
# @Software: PyCharm

import collections

# 使用collections模块，可以把它理解为一个容器，里面提供Python标准内建容器 dict , list , set , 和 tuple 的替代选择。

recordingClass = ["0~18", "19~38", "39~52", "53~60", "61~72", "73~77", "78~92"]

"""
["0~18","19~38","39~52","53~60","61~72","73~77","78~92"]
"""

locationname = collections.defaultdict(None, {
    "0": "cologne_butzweiler",
    "1": "cologne_fortiib",
    "2": "aachen_brand",
    "3": "bergheim_roemer",
    "4": "cologne_klettenberg",
    "5": "aachen_laurensberg",
    "6": "merzenich_rather",
})

lanlet2data = {
    # location 2
    "2": {
        # 车道1
        "-1": [1488, 1492, 1498, 1501, 1504, 1508],
        # 车道2
        "-2": [1489, 1493, 1499, 1502, 1574, 1509],
        # 车道3
        "-3": [],
        # 加速汇入车道
        "entry": [1500, 1503, 1566, 1567],
        # 上匝道
        "onramp": [1494, 1495],
        # 下匝道
        "exit": [1544, 1484, 1540],

        # 为什么选择lanelet2ID=1536的区块，那个地方应该是没有车经过的，除非有车非法进入。
        # 上游区域
        "mainlineUpstream": [1493, 1536, 1489, 1490],
        "area1": [1500],
        "area2": [1503],
        "area3": [1567],
        "area4": [1489, 1493, 1499],
        "area5": [1502, 1574],
        "area23": [1503, 1566, 1567],
        "area123": [1500, 1503, 1566, 1567],
        "length": {
            '1492': 29.72,
            '1493': 29.78,
            '1498': 67.92,
            '1499': 67.87,
            '1500': 67.56,
            '1501': 119.15,
            '1502': 119.45,
            '1489': 81.72,
            '1574': 40.65,
            "1567": 40.58,
            "1503": 119.67
        },
        "rightroadborder":[1510]

    },

    # location 3
    "3": {
        "-1": [1402, 1475, 1404, 1411, 1413, 1523, 1527, 1421],
        "-2": [1403, 1474, 1405, 1412, 1414, 1524, 1528, 1422],
        "-3": [],
        "entry": [1415, 1525, 1531],
        "onramp": [1498, 1408, 1476, 1479],
        "exit": [1445, 1447, 1451],
        "mainlineUpstream": [1412, 1478, 1405, 1406, 1404],
        "area1": [1415],
        "area2": [1525],
        "area3": [1531],
        "area4": [1405, 1412, 1414],
        "area5": [1524, 1528],
        "area23": [1525, 1530, 1531, 1528],
        "area123": [1415, 1525, 1530, 1531, 1528],
        "length": {
            '1415': 17.9,
            '1414': 17.9,
            '1422': 214.23,
            '1524': 168.03,
            '1525': 167.54,
            '1527': 32.33,
            '1528': 32.49,
            '1530': 32.92,
            "1405": 90.78,
            "1412": 25.86,
            "1425": 25.86,
            "1531": 32.87
        },
        "rightroadborder": [1530, 1523]

    },

    # location 5
    "5": {
        "-1": [1479, 1476, 1401, 1446, 1451, 1407, 1410, 1413, 1417],
        "-2": [1480, 1477, 1473, 1447, 1450, 1408, 1411, 1414, 1418],
        "-3": [1481, 1478, 1475],
        "entry": [1409, 1412, 1483],
        "onramp": [1405, 1448],
        "exit": [],
        "mainlineUpstream": [1499, 1450, 1446, 1447, 1473, 1474, 1475],
        "area1": [1409],
        "area2": [1412],
        "area3": [1483],
        "area4": [1447, 1450, 1408],
        "area5": [1411, 1414],
        "area23": [1412, 1482, 1483],
        "area123": [1409, 1412, 1482, 1483],
        "length": {
            '1408': 66.05,
            '1409': 66.54,
            '1411': 134.39,
            '1412': 132.63,
            "1414": 42.19,
            "1447": 31.9,
            "1450": 57.93,
            "1483": 42.21,
        },
        "rightroadborder": [1419]

    },

    # location 6
    "6": {
        "-1": [1451, 1452, 1457, 1461, 1465, 1470],
        "-2": [1450, 1453, 1458, 1462, 1466, 1471],
        "-3": [1449, 1454, 1459, 1463, 1467, 1472],
        "entry": [1460, 1514, 1513],
        "onramp": [1447, 1455],
        "exit": [],
        "mainlineUpstream": [1453, 1454, 1448, 1449],
        "area1": [1460],
        "area2": [1514],
        "area3": [1513],
        "area4": [1459, 1454],
        "area5": [1463, 1467],
        "area23": [1514, 1512, 1513],
        "area123": [1460, 1514, 1512, 1513],
        "length": {
            '1459': 26.91,
            '1460': 26.62,
            '1463': 191.65,
            '1514': 192.32,
            '1467': 27.4,
            '1454': 43.13,
            "1513": 27.71,
        },
        "rightroadborder": [1473]

    },
}