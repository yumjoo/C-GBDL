CUDA_VISIBLE_DEVICES=2,3 python train.py  --name bayes_AtriaSeg_16_ --train_txt ./train_AtriaSeg.txt --val_txt ./test_AtriaSeg.txt --label_factor_semi 0.2 --epochs 200
