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


def createArgs():
    # 参数容器
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")

    cs.add_argument('--distance_threshold', default=200,
                    help="distance threshold to match the surrounding vehicles.", type=int)

    cs.add_argument('--lookback', default=3,
                    help="this variable is set is to ensure the accuracy of the extracted trajectory.",
                    type=int)

    cs.add_argument('--timestep', default=0.04,
                    help="recording frequency.",
                    type=int)

    cs.add_argument('--location_set', default=['2', '3', '5', '6'],
                    help="location id set", type=list)

    return vars(cs.parse_args())


def main():
    # 日志文件记录
    config = createArgs()
    logger.info("Extracting trajectories and calculating metrics")
    logger.info("distance threshold {}, lookback is {}", config["distance_threshold"], config["lookback"])

    # 调用mergingextraction
    trajectoryExtraction = MergingExtractionClass(config)
    trajectoryExtraction.run()


if __name__ == '__main__':
    main()
