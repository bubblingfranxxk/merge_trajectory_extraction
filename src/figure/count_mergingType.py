# -*- coding = utf-8 -*-
# @Time : 2024/12/15 21:08
# @Author : 王砚轩
# @File : count_mergingType.py
# @Software: PyCharm

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from loguru import logger


def count_merging_type_in_csv(folder_path, column_name="MergingType"):
    """
    Counts the occurrences of each type in the specified column across multiple CSV files.

    Parameters:
        folder_path (str): The path to the folder containing the CSV files.
        column_name (str): The column name to analyze. Defaults to "mergingType".

    Returns:
        dict: A dictionary with merging types as keys and their counts as values.
    """
    merging_type_counter = Counter()

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the column exists
                if column_name in df.columns:
                    # Add 1 for each unique mergingType in this file
                    unique_types = df[column_name].unique()
                    merging_type_counter.update({merging_type: 1 for merging_type in unique_types})
                else:
                    logger.warning(f"Column '{column_name}' not found in file: {file_name}")
            except Exception as e:
                logger.error(f"Error reading file {file_name}: {e}")

    return dict(merging_type_counter)

if __name__ == '__main__':
    # Example usage
    rootPath = os.path.abspath('../../')
    assetPath = rootPath + "/asset/"
    folder_path = assetPath + '/single_traj/'  # Replace with the path to your folder containing the CSV files
    merging_type_counts = count_merging_type_in_csv(folder_path)

    # Print the results
    for merging_type, count in merging_type_counts.items():
        logger.info(f"{merging_type}: {count}")

    # Plot the results
    if merging_type_counts:
        sorted_types = sorted(merging_type_counts.keys())  # Sort types alphabetically
        sorted_counts = [merging_type_counts[mt] for mt in sorted_types]

        # Assign distinct colors
        colors = plt.cm.tab10(range(len(sorted_types)))

        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_types, sorted_counts, color=colors)

        # Add numbers above the bars
        for bar, count in zip(bars, sorted_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(count),
                     ha='center', va='bottom', fontsize=16)

        # Add legend
        plt.legend(bars, sorted_types, title="Merging Types", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xlabel('Merging Type')
        plt.ylabel('Count')
        plt.title('Merging Type Counts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(assetPath + '/count_mergingtype.png')
