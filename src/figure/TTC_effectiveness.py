# -*- coding = utf-8 -*-
# @Time : 2024/6/2 14:11
# @Author : 王砚轩
# @File : TTC_effectiveness.py
# @Software: PyCharm
import os
from TTC_acc_figure import create_output_folder
from TTC_acc_figure import dtype_spec
from loguru import logger
import pandas as pd


def main():
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    # 获取文件夹中合并的MergingTrajectory
    traj_files = [f for f in os.listdir(assetPath) if f.endswith('.csv')]
    # 建立检验的文件夹
    outpath = assetPath + "/Effectiveness/"
    create_output_folder(assetPath, "Effectiveness")

    for file in traj_files:
        if "Trajectory" not in file:
            continue
        file_path = assetPath + file
        traj = pd.read_csv(file_path, dtype=dtype_spec)
        logger.info("Loading {}", file)

        # 有效数据拼接
        ttc1Rear = traj[traj['RearTTCRaw1'] != 999]['RearTTCRaw1']
        ttc2Rear = traj[(traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1)]['RearTTCRaw2']
        ttc3Rear = traj[(traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw3']
        ttc1Lead = traj[traj['LeadTTCRaw1'] != 999]['LeadTTCRaw1']
        ttc2Lead = traj[(traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1)]['LeadTTCRaw2']
        ttc3Lead = traj[(traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['LeadTTCRaw3']
        ttc1LeftRear = traj[traj['LeftRearTTCRaw1'] != 999]['LeftRearTTCRaw1']
        ttc2LeftRear = traj[(traj['LeftRearTTCRaw2'] != 999) & (traj['LeftRearTTCRaw2'] != -1)]['LeftRearTTCRaw2']
        ttc3LeftRear = traj[(traj['LeftRearTTCRaw3'] > 0) & (traj['LeftRearTTCRaw3'] != 999)]['LeftRearTTCRaw3']
        ttc1LeftLead = traj[traj['LeftLeadTTCRaw1'] != 999]['LeftLeadTTCRaw1']
        ttc2LeftLead = traj[(traj['LeftLeadTTCRaw2'] != 999) & (traj['LeftLeadTTCRaw2'] != -1)]['LeftLeadTTCRaw2']
        ttc3LeftLead = traj[(traj['LeftLeadTTCRaw3'] > 0) & (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw3']
        ttc1LeftAlongside = traj[traj['LeftAlongsideTTCRaw1'] != 999]['LeftAlongsideTTCRaw1']
        ttc2LeftAlongside = \
            traj[(traj['LeftAlongsideTTCRaw2'] != 999) & (traj['LeftAlongsideTTCRaw2'] != -1)]['LeftAlongsideTTCRaw2']
        ttc3LeftAlongside = \
            traj[(traj['LeftAlongsideTTCRaw3'] > 0) & (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw3']

        ttc1 = pd.concat([ttc1Rear, ttc1Lead, ttc1LeftRear, ttc1LeftLead, ttc1LeftAlongside], ignore_index=True)
        ttc2 = pd.concat([ttc2Rear, ttc2Lead, ttc2LeftRear, ttc2LeftLead, ttc2LeftAlongside], ignore_index=True)
        ttc3 = pd.concat([ttc3Rear, ttc3Lead, ttc3LeftRear, ttc3LeftLead, ttc3LeftAlongside], ignore_index=True)

        # 失效情况数据 case1: tt1有效，ttc2失效，ttc3有效，case2: tt1有效，ttc2有效，ttc3失效，
        # case3:tt1有效，ttc2失效，ttc3失效，case4:tt1有效，ttc2有效，ttc3有效
        # case5: tt1失效，ttc2失效，ttc3有效，case6: tt1失效，ttc2有效，ttc3失效，
        # case7:tt1失效，ttc2失效，ttc3失效，case8:tt1失效，ttc2有效，ttc3有效
        case1Rear = traj[(traj['RearTTCRaw1'] != 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] == -1) &
                         (traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw3']
        case1Lead = traj[(traj['LeadTTCRaw1'] != 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] == -1) &
                         (traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['LeadTTCRaw3']
        case1LeftRear = traj[(traj['LeftRearTTCRaw1'] != 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] == -1) & (traj['LeftRearTTCRaw3'] > 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['LeftRearTTCRaw3']
        case1LeftLead = traj[(traj['LeftLeadTTCRaw1'] != 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] == -1) & (traj['LeftLeadTTCRaw3'] > 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw3']
        case1LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] != 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] == -1) & (traj['LeftAlongsideTTCRaw3'] > 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw3']

        case2Rear = traj[(traj['RearTTCRaw1'] != 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1) &
                         (traj['RearTTCRaw3'] < 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw2']
        case2Lead = traj[(traj['LeadTTCRaw1'] != 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1) &
                         (traj['LeadTTCRaw3'] < 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw2']
        case2LeftRear = traj[(traj['LeftRearTTCRaw1'] != 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] != -1) & (traj['LeftRearTTCRaw3'] < 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw2']
        case2LeftLead = traj[(traj['LeftLeadTTCRaw1'] != 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] != -1) & (traj['LeftLeadTTCRaw3'] < 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw2']
        case2LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] != 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] != -1) & (traj['LeftAlongsideTTCRaw3'] < 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw2']

        case3Rear = traj[(traj['RearTTCRaw1'] != 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] == -1) &
                         (traj['RearTTCRaw3'] < 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw1']
        case3Lead = traj[(traj['LeadTTCRaw1'] != 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] == -1) &
                         (traj['LeadTTCRaw3'] < 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw1']
        case3LeftRear = traj[(traj['LeftRearTTCRaw1'] != 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] == -1) & (traj['LeftRearTTCRaw3'] < 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw1']
        case3LeftLead = traj[(traj['LeftLeadTTCRaw1'] != 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] == -1) & (traj['LeftLeadTTCRaw3'] < 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw1']
        case3LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] != 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] == -1) & (traj['LeftAlongsideTTCRaw3'] < 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw1']

        case4Rear = traj[(traj['RearTTCRaw1'] != 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1) &
                         (traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw2']
        case4Lead = traj[(traj['LeadTTCRaw1'] != 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1) &
                         (traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw2']
        case4LeftRear = traj[(traj['LeftRearTTCRaw1'] != 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] != -1) & (traj['LeftRearTTCRaw3'] > 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw2']
        case4LeftLead = traj[(traj['LeftLeadTTCRaw1'] != 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] != -1) & (traj['LeftLeadTTCRaw3'] > 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw2']
        case4LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] != 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] != -1) & (traj['LeftAlongsideTTCRaw3'] > 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw2']

        case5Rear = traj[(traj['RearTTCRaw1'] == 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] == -1) &
                         (traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw3']
        case5Lead = traj[(traj['LeadTTCRaw1'] == 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] == -1) &
                         (traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['LeadTTCRaw3']
        case5LeftRear = traj[(traj['LeftRearTTCRaw1'] == 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] == -1) & (traj['LeftRearTTCRaw3'] > 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['LeftRearTTCRaw3']
        case5LeftLead = traj[(traj['LeftLeadTTCRaw1'] == 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] == -1) & (traj['LeftLeadTTCRaw3'] > 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw3']
        case5LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] == 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] == -1) & (traj['LeftAlongsideTTCRaw3'] > 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw3']

        case6Rear = traj[(traj['RearTTCRaw1'] == 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1) &
                         (traj['RearTTCRaw3'] < 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw2']
        case6Lead = traj[(traj['LeadTTCRaw1'] == 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1) &
                         (traj['LeadTTCRaw3'] < 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw2']
        case6LeftRear = traj[(traj['LeftRearTTCRaw1'] == 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] != -1) & (traj['LeftRearTTCRaw3'] < 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw2']
        case6LeftLead = traj[(traj['LeftLeadTTCRaw1'] == 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] != -1) & (traj['LeftLeadTTCRaw3'] < 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw2']
        case6LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] == 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] != -1) & (traj['LeftAlongsideTTCRaw3'] < 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw2']

        case7Rear = traj[(traj['RearTTCRaw1'] == 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] == -1) &
                         (traj['RearTTCRaw3'] < 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw1']
        case7Lead = traj[(traj['LeadTTCRaw1'] == 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] == -1) &
                         (traj['LeadTTCRaw3'] < 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw1']
        case7LeftRear = traj[(traj['LeftRearTTCRaw1'] == 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] == -1) & (traj['LeftRearTTCRaw3'] < 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw1']
        case7LeftLead = traj[(traj['LeftLeadTTCRaw1'] == 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] == -1) & (traj['LeftLeadTTCRaw3'] < 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw1']
        case7LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] == 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] == -1) & (traj['LeftAlongsideTTCRaw3'] < 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw1']

        case8Rear = traj[(traj['RearTTCRaw1'] == 999) & (traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1) &
                         (traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw2']
        case8Lead = traj[(traj['LeadTTCRaw1'] == 999) & (traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1) &
                         (traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['RearTTCRaw2']
        case8LeftRear = traj[(traj['LeftRearTTCRaw1'] == 999) & (traj['LeftRearTTCRaw2'] != 999) &
                             (traj['LeftRearTTCRaw2'] != -1) & (traj['LeftRearTTCRaw3'] > 0) &
                             (traj['LeftRearTTCRaw3'] != 999)]['RearTTCRaw2']
        case8LeftLead = traj[(traj['LeftLeadTTCRaw1'] == 999) & (traj['LeftLeadTTCRaw2'] != 999) &
                             (traj['LeftLeadTTCRaw2'] != -1) & (traj['LeftLeadTTCRaw3'] > 0) &
                             (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw2']
        case8LeftAlongside = traj[(traj['LeftAlongsideTTCRaw1'] == 999) & (traj['LeftAlongsideTTCRaw2'] != 999) &
                                  (traj['LeftAlongsideTTCRaw2'] != -1) & (traj['LeftAlongsideTTCRaw3'] > 0) &
                                  (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw2']

        case1 = pd.concat([case1Rear, case1Lead, case1LeftRear, case1LeftLead, case1LeftAlongside],
                          ignore_index=True)
        case2 = pd.concat([case2Rear, case2Lead, case2LeftRear, case2LeftLead, case2LeftAlongside],
                          ignore_index=True)
        case3 = pd.concat([case3Rear, case3Lead, case3LeftRear, case3LeftLead, case3LeftAlongside],
                          ignore_index=True)
        case4 = pd.concat([case4Rear, case4Lead, case4LeftRear, case4LeftLead, case4LeftAlongside],
                          ignore_index=True)
        case5 = pd.concat([case5Rear, case5Lead, case5LeftRear, case5LeftLead, case5LeftAlongside],
                          ignore_index=True)
        case6 = pd.concat([case6Rear, case6Lead, case6LeftRear, case6LeftLead, case6LeftAlongside],
                          ignore_index=True)
        case7 = pd.concat([case7Rear, case7Lead, case7LeftRear, case7LeftLead, case7LeftAlongside],
                          ignore_index=True)
        case8 = pd.concat([case8Rear, case8Lead, case8LeftRear, case8LeftLead, case8LeftAlongside],
                          ignore_index=True)

        result = {
            "ttc1": len(ttc1),
            "ttc2": len(ttc2),
            "ttc3": len(ttc3),
            "case1": len(case1),
            "case2": len(case2),
            "case3": len(case3),
            "case4": len(case4),
            "case5": len(case5),
            "case6": len(case6),
            "case7": len(case7),
            "case8": len(case8)
        }

        output_result = pd.DataFrame(data=result, index=[0])
        output_result.to_csv(outpath + r"/result.csv")


if __name__ == '__main__':
    main()
