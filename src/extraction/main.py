# -*- coding = utf-8 -*-
# @Time : 2023/12/19 15:35
# @Author : 王砚轩
# @File : main.py
# @Software: PyCharm

import pandas as pd
import numpy as np

import argparse
from loguru import logger

from src.extraction.mergingExtractionClass import MergingExtractionClass
from src.extraction.mergingPointClass import MergingPointClass
from src.extraction.matchMergingScenarioClass import matchScenariosClass


def createArgs():
    # 参数容器
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")

    cs.add_argument('--distance_threshold', default=200,
                    help="distance threshold to match the surrounding vehicles.",
                    type=int)

    cs.add_argument('--lookback', default=3,
                    help="this variable is set is to ensure the accuracy of the extracted trajectory.",
                    type=int)

    cs.add_argument('--timestep', default=0.04,
                    help="recording frequency.",
                    type=int)

    cs.add_argument('--location_set', default=['2', '3', '5', '6'],
                    help="location id set",
                    type=list)

    cs.add_argument('--save_mode', default='test',
                    help="-test means output document in output, -release means output document in result",
                    type=str)

    return vars(cs.parse_args())


def main():
    # 日志文件记录
    config = createArgs()
    logger.info("Extracting trajectories and calculating metrics")
    logger.info("distance threshold {}, lookback is {}", config["distance_threshold"], config["lookback"])

    # 调用mergingextraction，进行轨迹提取
    # trajectoryExtraction = MergingExtractionClass(config)
    # trajectoryExtraction.run()

    # 调用MergingPoint，提取合并点，并划分合并场景
    # pointExtraction = MergingPointClass(config)
    # pointExtraction.run()

    # 调用matchMergingScenarios，将汇入轨迹与汇入场景进行匹配
    matchScenarios = matchScenariosClass(config)
    matchScenarios.run()


if __name__ == '__main__':
    main()
