#!/bin/bash
while true
do
    # 检查0号GPU和1号GPU的显存使用情况
    free_memory_0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    free_memory_1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
    
    echo $free_memory_0 $free_memory_1
    # 判断显存是否足够
    if [ "$free_memory_1" -ge 25000 ]; then
        # 运行程序
        python train.py  --name bayes_AtriaSeg_16 --train_txt ./train_AtriaSeg.txt --val_txt ./test_AtriaSeg.txt --label_factor_semi 0.2 --epochs 240
        # 退出循环
        break
    else
        # 显存不足，等待10秒继续检测
        sleep 10
    fi
done
