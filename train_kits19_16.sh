CUDA_VISIBLE_DEVICES=0,1 python train.py  --name bayes_Kits19Seg_4_2 --train_txt ./train_kits19.txt --val_txt ./test_kits19.txt --label_factor_semi 0.025 --epochs 200
