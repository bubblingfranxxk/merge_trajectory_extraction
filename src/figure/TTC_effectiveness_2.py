# -*- coding = utf-8 -*-
# @Time : 2024/7/31 10:54
# @Author : 王砚轩
# @File : TTC_effectiveness_2.py
# @Software: PyCharm

import os
from TTC_acc_figure import create_output_folder
from TTC_acc_figure import dtype_spec
from loguru import logger
import pandas as pd


def main():
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    single_traj_path = assetPath + "/single_traj/"
    MR1, ML1, AR1, AL1 = 0, 0, 0, 0
    MR2, ML2, AR2, AL2 = 0, 0, 0, 0
    MR3, ML3, AR3, AL3 = 0, 0, 0, 0
    # 建立检验的文件夹
    outpath = assetPath + "/Effectiveness/"
    # 获取单条轨迹数据
    for f in os.listdir(single_traj_path):
        if not f.endswith('.csv'):
            continue
        file_path = os.path.join(single_traj_path, f)
        single_traj = pd.read_csv(file_path, dtype=dtype_spec)
        # 汇入帧
        merge_index = single_traj[single_traj['laneChange']].index[0]
        # 统计有效性
        MR1 = MR1 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftRearTTCRaw1'] != 999)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['RearTTCRaw1'] != 999)])
        MR2 = MR2 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftRearTTCRaw2'] != 999) &
                            (single_traj['LeftRearTTCRaw2'] != -1)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['RearTTCRaw2'] != 999) &
                            (single_traj['RearTTCRaw2'] != -1)])
        MR3 = MR3 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftRearTTCRaw3'] != 999) &
                            (single_traj['LeftRearTTCRaw3'] > 0)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['RearTTCRaw3'] != 999) &
                            (single_traj['RearTTCRaw3'] > 0)])

        ML1 = ML1 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftLeadTTCRaw1'] != 999)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['LeadTTCRaw1'] != 999)])
        ML2 = ML2 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftLeadTTCRaw2'] != 999) &
                            (single_traj['LeftLeadTTCRaw2'] != -1)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['LeadTTCRaw2'] != 999) &
                            (single_traj['LeadTTCRaw2'] != -1)])
        ML3 = ML3 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeftLeadTTCRaw3'] != 999) &
                            (single_traj['LeftLeadTTCRaw3'] > 0)]) + \
            len(single_traj[(single_traj.index >= merge_index) & (single_traj['LeadTTCRaw3'] != 999) &
                            (single_traj['LeadTTCRaw3'] > 0)])

        AR1 = AR1 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['RearTTCRaw1'] != 999)])
        AR2 = AR2 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['RearTTCRaw2'] != 999) &
                            (single_traj['RearTTCRaw2'] != -1)])
        AR3 = AR3 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['RearTTCRaw3'] != 999) &
                            (single_traj['RearTTCRaw3'] > 0)])

        AL1 = AL1 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeadTTCRaw1'] != 999)])
        AL2 = AL2 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeadTTCRaw2'] != 999) &
                            (single_traj['LeadTTCRaw2'] != -1)])
        AL3 = AL3 + \
            len(single_traj[(single_traj.index < merge_index) & (single_traj['LeadTTCRaw3'] != 999) &
                            (single_traj['LeadTTCRaw3'] > 0)])

        logger.info(f"recording id: {single_traj['recordingId'].values[0]}, "
                    f"track id: {single_traj['trackId'].values[0]} is finished.")

    result = {
        "MR1": MR1,
        "MR2": MR2,
        "MR3": MR3,
        "ML1": ML1,
        "ML2": ML2,
        "ML3": ML3,
        "AR1": AR1,
        "AR2": AR2,
        "AR3": AR3,
        "AL1": AL1,
        "AL2": AL2,
        "AL3": AL3
    }

    output_result = pd.DataFrame(data=result, index=[0])
    output_result.to_csv(outpath + r"/result2.csv")


if __name__ == '__main__':
    main()
