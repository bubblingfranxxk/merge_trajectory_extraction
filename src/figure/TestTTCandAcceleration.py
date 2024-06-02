# -*- coding = utf-8 -*-
# @Time : 2024/5/30 14:18
# @Author : 王砚轩
# @File : TestTTCandAcceleration.py
# @Software: PyCharm

from TTC_acc_figure import create_output_folder
from TTC_acc_figure import dtype_spec
from loguru import logger
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def Test(traj, partial=True):
    p1, p2, p3, s1, s2, s3, d1, d2, d3 = 999, 999, 999, 999, 999, 999, 999, 999, 999
    # TTC1 != 999, TTC2 != 999 and -1, TTC3 >0 and != 999
    ttc1Rear = traj[traj['RearTTCRaw1'] != 999]['RearTTCRaw1']
    ttc2Rear = traj[(traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1)]['RearTTCRaw2']
    ttc3Rear = traj[(traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['RearTTCRaw3']
    acc1Rear = traj[traj['RearTTCRaw1'] != 999]['lonAcceleration']
    acc2Rear = traj[(traj['RearTTCRaw2'] != 999) & (traj['RearTTCRaw2'] != -1)]['lonAcceleration']
    acc3Rear = traj[(traj['RearTTCRaw3'] > 0) & (traj['RearTTCRaw3'] != 999)]['lonAcceleration']

    ttc1Lead = traj[traj['LeadTTCRaw1'] != 999]['LeadTTCRaw1']
    ttc2Lead = traj[(traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1)]['LeadTTCRaw2']
    ttc3Lead = traj[(traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['LeadTTCRaw3']
    acc1Lead = traj[traj['LeadTTCRaw1'] != 999]['lonAcceleration']
    acc2Lead = traj[(traj['LeadTTCRaw2'] != 999) & (traj['LeadTTCRaw2'] != -1)]['lonAcceleration']
    acc3Lead = traj[(traj['LeadTTCRaw3'] > 0) & (traj['LeadTTCRaw3'] != 999)]['lonAcceleration']

    ttc1LeftRear = traj[traj['LeftRearTTCRaw1'] != 999]['LeftRearTTCRaw1']
    ttc2LeftRear = traj[(traj['LeftRearTTCRaw2'] != 999) & (traj['LeftRearTTCRaw2'] != -1)]['LeftRearTTCRaw2']
    ttc3LeftRear = traj[(traj['LeftRearTTCRaw3'] > 0) & (traj['LeftRearTTCRaw3'] != 999)]['LeftRearTTCRaw3']
    acc1LeftRear = traj[traj['LeftRearTTCRaw1'] != 999]['lonAcceleration']
    acc2LeftRear = traj[(traj['LeftRearTTCRaw2'] != 999) & (traj['LeftRearTTCRaw2'] != -1)]['lonAcceleration']
    acc3LeftRear = traj[(traj['LeftRearTTCRaw3'] > 0) & (traj['LeftRearTTCRaw3'] != 999)]['lonAcceleration']

    ttc1LeftLead = traj[traj['LeftLeadTTCRaw1'] != 999]['LeftLeadTTCRaw1']
    ttc2LeftLead = traj[(traj['LeftLeadTTCRaw2'] != 999) & (traj['LeftLeadTTCRaw2'] != -1)]['LeftLeadTTCRaw2']
    ttc3LeftLead = traj[(traj['LeftLeadTTCRaw3'] > 0) & (traj['LeftLeadTTCRaw3'] != 999)]['LeftLeadTTCRaw3']
    acc1LeftLead = traj[traj['LeftLeadTTCRaw1'] != 999]['lonAcceleration']
    acc2LeftLead = traj[(traj['LeftLeadTTCRaw2'] != 999) & (traj['LeftLeadTTCRaw2'] != -1)]['lonAcceleration']
    acc3LeftLead = traj[(traj['LeftLeadTTCRaw3'] > 0) & (traj['LeftLeadTTCRaw3'] != 999)]['lonAcceleration']

    ttc1LeftAlongside = traj[traj['LeftAlongsideTTCRaw1'] != 999]['LeftAlongsideTTCRaw1']
    ttc2LeftAlongside = \
        traj[(traj['LeftAlongsideTTCRaw2'] != 999) & (traj['LeftAlongsideTTCRaw2'] != -1)]['LeftAlongsideTTCRaw2']
    ttc3LeftAlongside = \
        traj[(traj['LeftAlongsideTTCRaw3'] > 0) & (traj['LeftAlongsideTTCRaw3'] != 999)]['LeftAlongsideTTCRaw3']
    acc1LeftAlongside = traj[traj['LeftAlongsideTTCRaw1'] != 999]['lonAcceleration']
    acc2LeftAlongside = \
        traj[(traj['LeftAlongsideTTCRaw2'] != 999) & (traj['LeftAlongsideTTCRaw2'] != -1)]['lonAcceleration']
    acc3LeftAlongside = \
        traj[(traj['LeftAlongsideTTCRaw3'] > 0) & (traj['LeftAlongsideTTCRaw3'] != 999)]['lonAcceleration']

    ttc1 = pd.concat([ttc1Rear, ttc1Lead, ttc1LeftRear, ttc1LeftLead, ttc1LeftAlongside], ignore_index=True)
    acc1 = pd.concat([acc1Rear, acc1Lead, acc1LeftRear, acc1LeftLead, acc1LeftAlongside], ignore_index=True)

    ttc2 = pd.concat([ttc2Rear, ttc2Lead, ttc2LeftRear, ttc2LeftLead, ttc2LeftAlongside], ignore_index=True)
    acc2 = pd.concat([acc2Rear, acc2Lead, acc2LeftRear, acc2LeftLead, acc2LeftAlongside], ignore_index=True)

    ttc3 = pd.concat([ttc3Rear, ttc3Lead, ttc3LeftRear, ttc3LeftLead, ttc3LeftAlongside], ignore_index=True)
    acc3 = pd.concat([acc3Rear, acc3Lead, acc3LeftRear, acc3LeftLead, acc3LeftAlongside], ignore_index=True)

    # Pearson和Spearman检验
    try:
        p1, p1_value = pearsonr(ttc1, acc1)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Pearson correlation for ttc1 and acc1: {e}, ttc1: {ttc1}, acc1: {acc1}")

    try:
        p2, p2_value = pearsonr(ttc2, acc2)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Pearson correlation for ttc2 and acc2: {e}, ttc2: {ttc2}, acc2: {acc2}")

    try:
        p3, p3_value = pearsonr(ttc3, acc3)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Pearson correlation for ttc3 and acc3: {e}, ttc3: {ttc3}, acc3: {acc3}")

    try:
        s1, s1_value = spearmanr(ttc1, acc1)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Spearman correlation for ttc1 and acc1: {e}, ttc1: {ttc1}, acc1: {acc1}")

    try:
        s2, s2_value = spearmanr(ttc2, acc2)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Spearman correlation for ttc2 and acc2: {e}, ttc2: {ttc2}, acc2: {acc2}")

    try:
        s3, s3_value = spearmanr(ttc3, acc3)
    except Exception as e:
        logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                       f"error track id: {traj['trackId'].values[0]}")
        logger.warning(f"Error calculating Spearman correlation for ttc3 and acc3: {e}, ttc3: {ttc3}, acc3: {acc3}")

    # 协整检验
    coint_test1 = sm.tsa.coint(ttc1, acc1)
    coint_test2 = sm.tsa.coint(ttc2, acc2)
    coint_test3 = sm.tsa.coint(ttc3, acc3)

    # 动态时间规整
    if partial:
        try:
            d1, path1 = fastdtw(ttc1, acc1, dist=euclidean)
        except Exception as e:
            logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                           f"error track id: {traj['trackId'].values[0]}")
            logger.warning(f"Error calculating DTW for ttc1 and acc1: {e}, ttc1: {ttc1}, acc1: {acc1}")

        try:
            d2, path2 = fastdtw(ttc2, acc2, dist=euclidean)
        except Exception as e:
            logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                           f"error track id: {traj['trackId'].values[0]}")
            logger.warning(f"Error calculating DTW for ttc2 and acc2: {e}, ttc2: {ttc2}, acc2: {acc2}")

        try:
            d3, path3 = fastdtw(ttc3, acc3, dist=euclidean)
        except Exception as e:
            logger.warning(f"Error recording id: {traj['recordingId'].values[0]}, "
                           f"error track id: {traj['trackId'].values[0]}")
            logger.warning(f"Error calculating DTW for ttc3 and acc3: {e}, ttc3: {ttc3}, acc3: {acc3}")

    return p1, p2, p3, s1, s2, s3, coint_test1, coint_test2, coint_test3, d1, d2, d3


def main():
    rootPath = os.path.abspath('../../')
    # 日志文件记录
    create_output_folder(rootPath, "log")
    logPath = rootPath + "/log/logfile.txt"
    logger.add(logPath, rotation="500 MB")
    assetPath = rootPath + "/asset/"
    # 获取文件夹中合并的MergingTrajectory
    traj_files = [f for f in os.listdir(assetPath) if f.endswith('.csv')]
    outpath = assetPath + "/Test/"
    # 建立检验的文件夹
    create_output_folder(assetPath, "Test")

    for file in traj_files:
        if "Trajectory" not in file:
            continue
        file_path = assetPath + file
        trajectory = pd.read_csv(file_path, dtype=dtype_spec)
        logger.info("Loading {}", file)

        total_p1, total_p2, total_p3, total_s1, total_s2,\
            total_s3, coint1, coint2, coint3, _, _, _ = Test(trajectory, partial=False)
        logger.info("TTC1 total info: Pearson Correlation: {}, Spearman Correlation: {}", total_p1, total_s1)
        logger.info("TTC2 total info: Pearson Correlation: {}, Spearman Correlation: {}", total_p2, total_s2)
        logger.info("TTC3 total info: Pearson Correlation: {}, Spearman Correlation: {}", total_p3, total_s3)
        logger.info("TTC1 total info: Coint Cointegration Test Statistic: {}, "
                    "Coint P-value: {}, Coint Critical Values: {}", coint1[0], coint1[1], coint1[2])
        logger.info("TTC2 total info: Coint Cointegration Test Statistic: {}, "
                    "Coint P-value: {}, Coint Critical Values: {}", coint2[0], coint2[1], coint2[2])
        logger.info("TTC3 total info: Coint Cointegration Test Statistic: {}, "
                    "Coint P-value: {}, Coint Critical Values: {}", coint3[0], coint3[1], coint3[2])
        # logger.info("TTC1 total info: DTW Distance: {}", total_d1)
        # logger.info("TTC2 total info: DTW Distance: {}", total_d2)
        # logger.info("TTC3 total info: DTW Distance: {}", total_d3)
        singleTraj = pd.DataFrame()
        coffeColumns = ["recordingId", "trackId", "Pearson1", "Pearson2", "Pearson3",
                        "Spearman1", "Spearman2", "Spearman3", "DTW_Distance1", "DTW_Distance2", "DTW_Distance3"]
        coffList = pd.DataFrame(columns=coffeColumns)
        for index, row in trajectory.iterrows():
            if not singleTraj.empty \
                    and (row['recordingId'] != singleTraj['recordingId'].values[0]
                         or row['trackId'] != singleTraj['trackId'].values[0]):
                logger.info("Current recording id is {}, current track id is {}", row['recordingId'], row['trackId'])
                p1, p2, p3, s1, s2, s3, _, _, _, d1, d2, d3 = Test(singleTraj)
                temp = {
                    "recordingId": row["recordingId"],
                    "trackId": row["trackId"],
                    "Pearson1": p1,
                    "Pearson2": p2,
                    "Pearson3": p3,
                    "Spearman1": s1,
                    "Spearman2": s2,
                    "Spearman3": s3,
                    "DTW_Distance1": d1,
                    "DTW_Distance2": d2,
                    "DTW_Distacne3": d3
                }
                coffList = pd.concat([coffList, pd.DataFrame(data=temp, index=[0])], ignore_index=True)
                temp = {}
                singleTraj = pd.DataFrame()
            newrow = pd.DataFrame(row).transpose()
            # print(newrow)
            singleTraj = pd.concat([singleTraj, newrow])

        # 保存
        coffList.to_csv(outpath + r"\Pearson_Spearman_coefficient.csv")


if __name__ == '__main__':
    main()
