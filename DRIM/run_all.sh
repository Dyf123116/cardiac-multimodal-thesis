#!/bin/bash
cd /home/featurize/DRIM
grep -q "from_cli" train_drim_cls.py || { echo "请先完成种子修复"; exit 1; }
while pgrep -f "general.seed=200[01]" > /dev/null; do sleep 60; done          # 等正在跑的训练结束
for s in 2000 2001; do                                                       # 没训练过的补上
  [ -f models/drim_seed${s}.pt ] || python train_drim_cls.py general.seed=$s > logs/train_seed${s}.log 2>&1
done
for s in 2000 2001; do                                                       # 推理时掩码评估
  C=./models/drim_seed${s}.pt
  DRIM_EVAL_CKPT=$C DRIM_TAG=_evalfull python train_drim_cls.py general.seed=$s > logs/eval_${s}_full.log 2>&1
  DRIM_EVAL_CKPT=$C DRIM_TAG=_maskecg DRIM_MASK_ECG=1 python train_drim_cls.py general.seed=$s > logs/eval_${s}_mask.log 2>&1
  DRIM_EVAL_CKPT=$C DRIM_TAG=_val_full DRIM_EVAL_SPLIT=val python train_drim_cls.py general.seed=$s > logs/eval_${s}_valfull.log 2>&1
  DRIM_EVAL_CKPT=$C DRIM_TAG=_val_maskecg DRIM_EVAL_SPLIT=val DRIM_MASK_ECG=1 python train_drim_cls.py general.seed=$s > logs/eval_${s}_valmask.log 2>&1
done
for s in 1999 2000 2001; do                                                  # 训练时模态消融
  DRIM_NO_ECG_TRAIN=1 DRIM_TAG=_noecg python train_drim_cls.py general.seed=$s > logs/train_noecg_${s}.log 2>&1
  DRIM_NO_ECG_TRAIN=1 DRIM_EVAL_CKPT=./models/drim_seed${s}_noecg.pt DRIM_EVAL_SPLIT=val DRIM_TAG=_noecg_val \
    python train_drim_cls.py general.seed=$s > logs/eval_noecg_${s}_val.log 2>&1
done
python gap.py > logs/gap_result.txt 2>&1
echo 全部完成 >> logs/gap_result.txt
