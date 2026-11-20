import numpy as np, pandas as pd, os, re, sys, ast
from sklearn.metrics import roc_auc_score
def macro(p, y):
    s = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)]
    return float(np.mean(s))
# 用索引表里是否有 ECG 编号来分组（与 10 月的分析一致）
m = re.search(r'^\s*OUT\s*=\s*(.+?)\s*$', open('train_drim_cls.py').read(), re.M)
csv = os.environ.get('CSV') or f"{ast.literal_eval(m.group(1))}/cardiac_drim.csv"
df = pd.read_csv(csv)
groups = {'测试': df[df.split == 'test'].study_id.notna().values, '验证': df[df.split == 'val'].study_id.notna().values}
rows = []
for seed in sys.argv[1:] or ['1999', '2000', '2001']:
    for split, full, mask, abl in [('测试', '_evalfull', '_maskecg', '_noecg'), ('验证', '_val_full', '_val_maskecg', '_noecg_val')]:
        f = lambda t: f'preds/drim_test_seed{seed}{t}.npz'
        if not (os.path.exists(f(full)) and os.path.exists(f(mask))):
            print(f'seed {seed} {split}: 缺少 {f(full)} 或 {f(mask)}'); continue
        a, b = np.load(f(full)), np.load(f(mask)); pf, pm, y = a['probs'], b['probs'], a['labels']
        has = groups[split]; assert len(has) == len(y), '索引表行数与预测不一致'
        same = np.abs(pf - pm).max(1) <= 1e-6
        e_full, e_mask, no = macro(pf[has], y[has]), macro(pm[has], y[has]), macro(pf[~has], y[~has])
        gap = e_full - no; fr_mask = (e_full - e_mask) / gap
        line = (f'seed {seed} {split}: 有ECG={e_full:.4f} 掩码后={e_mask:.4f} 无ECG={no:.4f} '
                f'(有ECG {has.sum()} / 无ECG {(~has).sum()}; 有编号但掩码前后预测不变 {int((has & same).sum())}) 推理时掩码占比={fr_mask:.1%}')
        fr_abl = np.nan
        if os.path.exists(f(abl)):
            pa = np.load(f(abl))['probs']; e_abl = macro(pa[has], y[has]); fr_abl = (e_full - e_abl) / gap
            line += f' | 不用ECG训练={e_abl:.4f} 训练时消融占比={fr_abl:.1%}'
        rows.append((split, fr_mask, fr_abl)); print(line)
for split in ('测试', '验证'):
    v = np.array([(a, b) for s, a, b in rows if s == split])
    if len(v) > 1:
        out = f'{split}（n={len(v)}）：推理时掩码 {np.mean(v[:,0]):.1%} ± {np.std(v[:,0], ddof=1):.1%}'
        w = v[~np.isnan(v[:,1]), 1]
        if len(w) > 1: out += f'；训练时消融 {np.mean(w):.1%} ± {np.std(w, ddof=1):.1%}（n={len(w)}）'
        print(out)
