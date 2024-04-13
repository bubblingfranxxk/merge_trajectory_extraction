#!/bin/bash

# 删除当前目录下所有__pycache__文件夹及其内部的文件
find . -type d -name "__pycache__" -exec rm -rf {} +