import os
import argparse
from copy import deepcopy
import pickle
from pathlib import Path
from argparse import Namespace

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models import DrFuseTrainer
from utils import get_ehr_datasets
from drfuse_data_adapter import ECGHdf5Dataset,ecg_collate_fn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_dir', type=str, default='./dataset/mimiciv_ehr_cxr')
    parser.add_argument('--hdf5_path', type=str, required=True,help='ecg_waveforms.h5 路径')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--missing_ratio', type=float, default=0.3,help='训练时随机丢弃 ECG 的比例')
    parser.add_argument('--timestep', type=float, default=1.0)
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--aug_missing_ratio', type=float, default=0.3)
    parser.add_argument('--ehr_feat_missing_ratio', type=float, default=0.0,
                        help='训练时 EHR 特征级随机缺失比例（NAIM 正则化）')
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--normalizer_state', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--text_emb_path',   type=str,
                        default='/home/featurize/data/output/text_embeddings.npy')
    parser.add_argument('--text_index_path', type=str,
                        default='/home/featurize/data/output/text_index.csv')


    args = parser.parse_args()


    # set number of threads allowed
    torch.set_num_threads(5)

    # set seed
    L.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_dir)   # 指向 ./output/drfuse_input/
    hdf5_path = Path(args.hdf5_path)  # 指向 ./output/ecg_waveforms.h5
    ehr_dir   = data_root
    aligned_csv = Path('/home/featurize/data/output/ecg_icu_aligned.csv')
    
    train_dataset = ECGHdf5Dataset(
        hdf5_path         = hdf5_path,
        ehr_pkl_path      = data_root / 'ehr_phenotyping_48h_train.pkl',
        aligned_csv       = aligned_csv,
        text_emb_path    = Path(args.text_emb_path),    # ← 新增
        text_index_path  = Path(args.text_index_path),
        aug_missing_ratio = args.missing_ratio,
        is_train          = True,
        ehr_feat_missing_ratio = args.ehr_feat_missing_ratio,
    )
    val_dataset = ECGHdf5Dataset(
        hdf5_path    = hdf5_path,
        ehr_pkl_path = data_root / 'ehr_phenotyping_48h_val.pkl',
        aligned_csv       = aligned_csv,
        text_emb_path    = Path(args.text_emb_path),    # ← 新增
        text_index_path  = Path(args.text_index_path),
    )
    test_dataset = ECGHdf5Dataset(
        hdf5_path    = hdf5_path,
        ehr_pkl_path = data_root / 'ehr_phenotyping_48h_test.pkl',
        aligned_csv       = aligned_csv,
        text_emb_path    = Path(args.text_emb_path),    # ← 新增
        text_index_path  = Path(args.text_index_path),
    )

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True,  num_workers=args.num_workers, pin_memory=True,collate_fn=ecg_collate_fn)
    val_dl   = DataLoader(val_dataset,   batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=ecg_collate_fn)
    test_dl  = DataLoader(test_dataset,  batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DrFuseTrainer(args=args, label_names=train_dl.dataset.CLASSES)

    callback_metric = 'val_PRAUC_avg_over_dxs/final'
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                        min_delta=0.00,
                                        patience=args.patience,
                                        verbose=False,
                                        mode="max")
    checkpoint_callback = ModelCheckpoint(monitor='val_PRAUC_avg_over_dxs/final',
                                          mode='max',
                                          save_top_k=1,
                                          filename='best',
                                          verbose=True,)


    #trainer = L.Trainer(devices=[0],
                        #accelerator='gpu',
                        #max_epochs=args.epochs,
                        #min_epochs=min(args.epochs, 10),
                        #callbacks=[early_stop_callback])
    trainer = L.Trainer(devices=[0],
                        accelerator='gpu',
                        max_epochs=args.epochs,
                        min_epochs=min(args.epochs, 10),
                        gradient_clip_val=1.0,
                        callbacks=[early_stop_callback, checkpoint_callback],
                       )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # do testing
    results = {
        'best_val_prauc': checkpoint_callback.best_model_score.item(),
        'best_val_roauc': trainer.callback_metrics['val_AUROC_avg_over_dxs/final'].item()
    }
    logpath = trainer.logger.log_dir
    trainer.loggers = None
    test_dl_partial = DataLoader(
        ECGHdf5Dataset(
            hdf5_path    = hdf5_path,
            ehr_pkl_path = data_root / 'ehr_phenotyping_48h_test.pkl',
            aligned_csv  = aligned_csv,
            text_emb_path   = Path(args.text_emb_path),
            text_index_path = Path(args.text_index_path),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=ecg_collate_fn,
    )
    test_dl_paired = DataLoader(
        ECGHdf5Dataset(
            hdf5_path    = hdf5_path,
            ehr_pkl_path = data_root / 'ehr_phenotyping_48h_test.pkl',
            aligned_csv  = aligned_csv,
            paired_only  = True,
            text_emb_path   = Path(args.text_emb_path),
            text_index_path = Path(args.text_index_path),
        ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=ecg_collate_fn,
    )

    trainer.test(model=model, dataloaders=test_dl_partial, ckpt_path='best')
    results['partial_test_results'] = deepcopy(model.test_results)

    trainer.test(model=model, dataloaders=test_dl_paired,  ckpt_path='best')
    results['paired_test_results'] = deepcopy(model.test_results)
    
    with open(logpath+'/test_results', 'wb') as f:
        pickle.dump(results, f)
    # 加这几行打印结果
    print("\n" + "="*50)
    print("训练结果:")
    print(f"  最佳 Val PRAUC: {checkpoint_callback.best_model_score.item():.4f}")
    print(f"  最佳 Val AUROC: {results['best_val_roauc']:.4f}")
    print("\nPartial 测试集结果（含无ECG患者）:")
    print(f"  PRAUC: {results['partial_test_results']['prauc']:.4f}")
    print(f"  AUROC: {results['partial_test_results']['auroc']:.4f}")
    print("\nPaired 测试集结果（仅有ECG患者）:")
    print(f"  PRAUC: {results['paired_test_results']['prauc']:.4f}")
    print(f"  AUROC: {results['paired_test_results']['auroc']:.4f}")
    print("="*50)