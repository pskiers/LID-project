import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve


def load_flipd_dataframe(json_path: str) -> pd.DataFrame:
    """
    Reads a JSON list of { prompt: { "flipd": int } } entries
    and returns a DataFrame with columns ["prompt","flipd"].
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for prompt, info in data.items():
        # each entry is a single‐key dict: {prompt: {...}}
        rows.append({
            "prompt": prompt,
            "flipd": info.get("flipd", float("nan"))
        })
    return pd.DataFrame(rows)


def get_lid_stat_df(path):
    df = load_flipd_dataframe(path)
    df['base_prompt'] = df['prompt'].str.replace(r'\s*\d+\s*$', '', regex=True)
    lid_stats = (
        df.groupby('base_prompt')['flipd']
        .agg(mean_val='mean', std_val='std')
        .reset_index()
    )
    return lid_stats


def compute_scores_and_metrics(df_real_pos, df_fake_neg, min_std=1e-6):
    # label and combine
    df_real = df_real_pos.copy()
    df_real['label'] = 1
    df_fake = df_fake_neg.copy()
    df_fake['label'] = 0
    df = pd.concat([df_real, df_fake], ignore_index=True)

    # ensure numeric and handle zero/near-zero std
    df['std_val_clipped'] = df['std_val'].astype(float).clip(lower=min_std)
    vals = df['flipd'].astype(float).to_numpy()
    mus  = df['mean_val'].astype(float).to_numpy()
    sds  = df['std_val_clipped'].to_numpy()

    # score = log pdf under the real distribution (higher => more likely real)
    # use scipy.stats.norm.logpdf (vectorized)
    scores = norm.logpdf(vals, loc=mus, scale=sds)
    df['score'] = scores

    # AUC (scores must be higher for positives)
    auc = roc_auc_score(df['label'].to_numpy(), df['score'])

    # find threshold that maximizes raw accuracy on this pooled dataset
    # use thresholds from ROC (they are returned sorted descending)
    fpr, tpr, thr = roc_curve(df['label'].to_numpy(), df['score'])
    best_acc = -1.0
    best_thr = None
    best_pred = None
    for t in thr:
        pred = (df['score'] >= t).astype(int)
        acc = accuracy_score(df['label'], pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = t
            best_pred = pred

    # metrics at best threshold
    precision, recall, f1, _ = precision_recall_fscore_support(df['label'], best_pred, average='binary', zero_division=0)
    cm = confusion_matrix(df['label'], best_pred)

    result = {
        'auc': auc,
        'best_accuracy': best_acc,
        'best_threshold': best_thr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'df_with_scores': df  # optional, for inspection
    }
    return result


def interval_classifier_tune(
    df_real_pos,
    df_fake_neg,
    thr_grid=None,
    min_std=1e-8,
    objective='accuracy',  # one of 'accuracy', 'f1', 'balanced_accuracy'
    return_all_scores=False
):
    """
    Fit and tune a simple interval classifier:
      predict = 1 if val in [mean - thr*std, mean + thr*std]

    Args:
      df_real_pos, df_fake_neg: DataFrames with columns ['flipd','mean_val','std_val'].
      thr_grid: iterable of thresholds to try (if None, uses np.linspace(0,5,501)).
      min_std: clip lower bound for std to avoid div-by-zero.
      objective: metric to maximize when choosing thr ('accuracy' (default), 'f1', 'balanced_accuracy').

    Returns:
      dict with keys:
        'auc' : ROC AUC using continuous score = -abs(z)
        'best_thr' : chosen thr
        'best_metrics' : dict(accuracy, precision, recall, f1)
        'confusion_matrix' : confusion matrix for best thr (rows=true 0/1, cols=pred 0/1)
        'results_df' : DataFrame with columns ['flipd','mean_val','std_val','z','score','label','pred_best_thr']
        'per_thr' (optional if return_all_scores=True): DataFrame of thr -> metrics
    """
    # combine
    df_r = df_real_pos.copy().assign(label=1)
    df_f = df_fake_neg.copy().assign(label=0)
    df = pd.concat([df_r, df_f], ignore_index=True)

    # ensure numeric and clip std
    df['real_std_clipped'] = df['std_val'].astype(float).clip(lower=min_std)
    df['flipd'] = df['flipd'].astype(float)
    df['mean_val'] = df['mean_val'].astype(float)

    # z-score and continuous score (higher => more likely real)
    df['z'] = (df['flipd'] - df['mean_val']) / df['real_std_clipped']
    # use -abs(z) as continuous score (closer to mean -> more likely real)
    df['score'] = -np.abs(df['z'])

    # AUC (works even if score ties)
    try:
        auc = roc_auc_score(df['label'].to_numpy(), df['score'].to_numpy())
    except ValueError:
        auc = float('nan')  # if only one class present, AUC undefined

    # thr grid
    if thr_grid is None:
        # pick grid relative to typical z-range
        # use 0..5 with fine steps by default
        thr_grid = np.linspace(0.0, 5.0, 501)

    # iterate and evaluate
    rows = []
    best_val = -np.inf
    best_thr = None
    best_pred = None
    for thr in thr_grid:
        preds = ((df['flipd'] >= (df['mean_val'] - thr*df['real_std_clipped'])) &
                 (df['flipd'] <= (df['mean_val'] + thr*df['real_std_clipped']))).astype(int)
        acc = accuracy_score(df['label'], preds)
        prec, rec, f1, _ = precision_recall_fscore_support(df['label'], preds, average='binary', zero_division=0)
        # compute balanced accuracy if requested
        tn, fp, fn, tp = confusion_matrix(df['label'], preds).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (tpr + tnr)

        rows.append({'thr': thr, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'balanced_accuracy': bal_acc})

        # objective selection
        if objective == 'accuracy':
            score_obj = acc
        elif objective == 'f1':
            score_obj = f1
        elif objective == 'balanced_accuracy':
            score_obj = bal_acc
        else:
            raise ValueError("objective must be one of 'accuracy','f1','balanced_accuracy'")

        if score_obj > best_val:
            best_val = score_obj
            best_thr = thr
            best_pred = preds.copy()

    per_thr_df = pd.DataFrame(rows)

    # metrics for best threshold
    prec, rec, f1, _ = precision_recall_fscore_support(df['label'], best_pred, average='binary', zero_division=0)
    acc = accuracy_score(df['label'], best_pred)
    cm = confusion_matrix(df['label'], best_pred)

    # attach best-pred to results df
    df = df.assign(pred_best_thr=best_pred)

    out = {
        'auc': auc,
        'best_thr': best_thr,
        'best_metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'confusion_matrix': cm,
        'results_df': df
    }
    if return_all_scores:
        out['per_thr'] = per_thr_df
    return out


stat_lid_for_real = "diffusion_memorization/outputs/attributions/real-10-seeds-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json"
stat_lid_for_fake = "diffusion_memorization/outputs/attributions/fake-10-seeds-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json"
real_lid = "diffusion_memorization/outputs/attributions/laion1024-lid-estimates-sdxl-conditional-ai_flipd.json"
fake_lid = "diffusion_memorization/outputs/attributions/sdxl-laionv2-lid-estimates-sdxl-conditional-ai_flipd.json"

def main():
    df_stat_real = get_lid_stat_df(stat_lid_for_real)
    df_stat_fake = get_lid_stat_df(stat_lid_for_fake)

    df_real = load_flipd_dataframe(real_lid)
    df_fake = load_flipd_dataframe(fake_lid)

    merged_real = df_real.merge(df_stat_real, left_on='prompt', right_on='base_prompt', how='left').dropna()
    merged_fake = df_fake.merge(df_stat_fake, left_on='prompt', right_on='base_prompt', how='left').dropna()

    results = compute_scores_and_metrics(merged_real, merged_fake)
    results.pop('df_with_scores')
    print("Gaussian", results)
    results_interval = interval_classifier_tune(merged_real, merged_fake)
    results_interval.pop('results_df')
    print("Interval", results)


if __name__ == "__main__":
    main()
