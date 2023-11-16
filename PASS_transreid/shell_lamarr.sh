#!/bin/bash

# 获取脚本的目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 改变工作目录到脚本所在的目录
cd "$DIR"

# sh xxx.sh trainlog
# 检查是否有参数传递给脚本
if [ "$#" -ne 1 ]; then
    echo "使用方法: $0 <日志名称>"
    exit 1
fi

# 获取日志名称参数
LOG_NAME=$1

# 设置日志文件夹名称（你可以根据需要修改）
LOG_FOLDER="shell_lamar_log"

# 创建日志文件夹，如果不存在
mkdir -p "$LOG_FOLDER"

# 获取当前时间并格式化为文件名
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="./${LOG_FOLDER}/${LOG_NAME}_${CURRENT_TIME}.txt"

# 使用nohup在后台执行训练命令，并将所有输出（包括错误输出）重定向到日志文件, 注意swin中的device_id必须在yml设置。
nohup python train.py \
--config_file configs/mars/vit_base.yml \
MODEL.PRETRAIN_PATH '../pretrainedlup_model/pass_vit_base_full.pth' \
OUTPUT_DIR './logs/mars/pass_vit_base_full_cat384' \
SOLVER.MAX_EPOCHS 120 \
SOLVER.CHECKPOINT_PERIOD 120 \
SOLVER.EVAL_PERIOD 30 \
SOLVER.IMS_PER_BATCH 16 \
DATALOADER.NUM_WORKERS 24 \
INPUT.SIZE_TRAIN "[384, 128]" \
INPUT.SIZE_TEST "[384, 128]" \
MODEL.DEVICE_ID "('0')" \
>> "$LOG_FILE" 2>&1 &


# 将脚本结果写入日志文件
echo "训练命令已在后台启动。训练日志文件已保存到：$LOG_FILE"
