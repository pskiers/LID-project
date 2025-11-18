# import json


# def load_results(file_path):
#     with open(file_path, 'r') as file:
#         results = json.load(file)
#     return results

# path = "lidy_sdxl_turbo.json"

# all_lids = []
# for prompt_with_dog, values_dog in load_results(path).items():
#     if values_dog["flipd"] == values_dog["flipd"]:
#         all_lids.append(values_dog["flipd"])
#     # if "dog" in prompt_with_dog.lower():
#     #     for prompt, values_no_dog in load_results(path).items():
#     #         if prompt.lower() in [
#     #             prompt_with_dog.lower().replace("dog", ""), 
#     #             prompt_with_dog.lower().replace("dog ", ""),
#     #             prompt_with_dog.lower().replace(" dog", ""),
#     #         ]:
#     #             print(prompt_with_dog)
#     #             print(values_dog["flipd"] - values_no_dog["flipd"])
# print(sum(all_lids) / len(all_lids))

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def best_threshold_accuracy(X, Y):
    # Convert to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert set(np.unique(Y)) <= {0, 1}, "Y must be binary (0/1)"

    # Unique sorted values in X
    vals = np.unique(X)
    # Candidate thresholds: midpoints + extremes
    thresholds = np.concatenate([
        [-np.inf],
        (vals[:-1] + vals[1:]) / 2,
        [np.inf]
    ])

    best_acc = 0.0
    best_t = thresholds[0]

    for t in thresholds:
        # Predict 1 if X >= t, else 0
        preds = (X >= t).astype(int)
        acc = accuracy_score(Y, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_acc

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

# # --- Load both files ---
# df1 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/laion1024-lid-estimates-sdxl-unconditional_flipd.json")
# df2 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/sdxl-laionv2-lid-estimates-sdxl-unconditional_flipd.json")

# # --- Aggregate per-prompt stats in each file ---
# agg1 = df1.groupby("prompt")["flipd"].agg(mean1="mean", std1="std")
# agg2 = df2.groupby("prompt")["flipd"].agg(mean2="mean", std2="std")

# # 2) Compute per‑prompt mean and std in each file
# agg1 = df1.groupby("prompt")["flipd"].agg(mean1="mean", std1="std")
# agg2 = df2.groupby("prompt")["flipd"].agg(mean2="mean", std2="std")

# # 3) Restrict to prompts present in both
# common = agg1.index.intersection(agg2.index)
# agg1_c = agg1.loc[common]
# agg2_c = agg2.loc[common]

# # 4) Merge side‑by‑side
# merged = pd.concat([agg1_c, agg2_c], axis=1)

# # 5) Compute overall averages across prompts
# overall_mean1    = merged["mean1"].mean()
# overall_std1     = merged["mean1"].std()
# overall_mean2    = merged["mean2"].mean()
# overall_std2     = merged["mean2"].std()
# overall_mean_diff= (merged["mean1"] - merged["mean2"]).mean()
# overall_std_diff= (merged["mean1"] - merged["mean2"]).std()

# # 6) Report
# print(f"Number of common prompts: {len(common)}")
# print(f"laion - flipd = {overall_mean1 + 65536:.3f} +- {overall_std1:.3f}")
# print(f"sdxl - flipd = {overall_mean2 + 65536:.3f} +- {overall_std2:.3f}")
# print(f"difference (laion -_sdxl): {overall_mean_diff:.3f} +- {overall_std_diff:.3f}")


# plt.figure()
# plt.hist(merged["mean1"] - merged["mean2"], bins=50, density=True, alpha=0.6, color='blue', label='Real Images - Generated Images')

# # Labels and legend
# plt.xlabel('PNG Length Ratio')
# plt.ylabel('Density')
# # plt.xlim(0, 1.5)
# plt.legend()

# plt.tight_layout()
# plt.show()





import json
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys


def main():
    # 1) Load both files
    df1 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/laion1024-lid-estimates-sdxl-unconditional_flipd.json")
    df2 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/sdxl-laionv2-lid-estimates-sdxl-unconditional_flipd.json")

    # 2) Find common prompts
    common = set(df1["prompt"]).intersection(df2["prompt"])
    if not common:
        print("No prompts in common. Exiting.")
        sys.exit(1)

    # 3) Filter to common prompts and label them
    df1_c = df1[df1["prompt"].isin(common) & df1["flipd"].notnull()].copy()
    df1_c["label"] = 0
    df2_c = df2[df2["prompt"].isin(common) & df2["flipd"].notnull()].copy()
    df2_c["label"] = 1

    # 4) Combine into one DataFrame
    combined = pd.concat([df1_c, df2_c], ignore_index=True)

    # 5) Compute AUC
    auc = roc_auc_score(combined["label"], combined["flipd"])
    print(f"ROC AUC (flipd → file label): {auc:.4f}")
    print(f"Accuracy (flipd → file label): {best_threshold_accuracy(combined['flipd'], combined['label']):.4f}")
    print()
    # (Optional) save combined dataset to CSV
    # combined.to_csv("flipd_combined.csv", index=False)
    # print("Saved combined dataset to flipd_combined.csv")

# if __name__ == "__main__":
#     main()



import json
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os
import textwrap

def load_json_as_df(path, value_key):
    """
    Loads a JSON of the form [{prompt: {value_key: int}}, ...]
    or { prompt: int, ... } into a DataFrame with columns:
      - 'prompt'
      - value_key (int)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []

    for prompt, val in data.items():
        rows.append({'prompt': prompt, value_key: val})
    return pd.DataFrame(rows)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def simple_binary_classifier(
    X,
    Y,
    test_size: float = 0.2,
    random_state: int = 42,
    model: str = "tree",  # "logistic" or "tree"
    max_depth: int = 2
):
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        y_arr = np.asarray(Y).ravel()
    else:
        y_arr = np.asarray(Y).ravel()

    # Ensure X is 2D: (n_samples, n_features)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 1D or 2D array-like, got shape {X_arr.shape}")

    # Ensure binary labels; encode if necessary
    uniq = np.unique(y_arr)
    if uniq.size == 1:
        raise ValueError("Y contains only a single class; need two classes for binary classification.")
    if uniq.size > 2:
        raise ValueError(f"Y contains more than two classes: {uniq}. This function is for binary classification.")

    # Map labels to 0/1 if not already numeric 0/1
    if not (set(uniq) <= {0, 1}):
        le = LabelEncoder()
        y_arr = le.fit_transform(y_arr)
    else:
        le = None  # no encoder used

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    # Choose model
    if model == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model == "tree":
        clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    else:
        raise ValueError("model must be 'logistic' or 'tree'")

    # Fit
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model": clf,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "accuracy": acc, "report": report, "confusion_matrix": cm,
        "label_encoder": le
    }

    # Print a short summary
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    return results

def main():
    # 1) Load DataFrames
    df1 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/laion1024-lid-estimates-sdxl-conditional-ai_flipd.json")
    df2 = load_flipd_dataframe("diffusion_memorization/outputs/attributions/sdxl-laionv2-ai-lid-estimates-sdxl-conditional_flipd.json")
    common = set(df1["prompt"]).intersection(df2["prompt"])
    df1 = df1[df1["prompt"].isin(common) & df1["flipd"].notnull()].copy()
    df2 = df2[df2["prompt"].isin(common) & df2["flipd"].notnull()].copy()
    extra1 = load_json_as_df("data/laion-2b-1024x1024/preds.json", 'value')
    extra2 = load_json_as_df("data/sdxl_samplesv2_ai_jpg/preds.json", 'value')

    # 2) Merge on prompt
    df1 = df1.merge(extra1, on='prompt', how='inner')  # inner ensures common prompts
    df2 = df2.merge(extra2, on='prompt', how='inner')
    df1 = df1.dropna(subset=["flipd", "value"])
    df2 = df2.dropna(subset=["flipd", "value"])

    # 3) Compute diff = flipd + 66536 - value
    for df in (df1, df2):
        df['diff'] = (df['flipd'] + 65536) / df['value']

    # 4) Concatenate and report mean & std of diff
    combined = pd.concat([df1.assign(label=0), df2.assign(label=1)], ignore_index=True)

    print(f"Total prompts used: {combined['prompt'].nunique()}")
    print(f"laion png length = {df1['value'].mean():.3f} +- {df1['value'].std(): .3f}")
    print(f"sdxl png length = {df2['value'].mean():.3f} +- {df2['value'].std(): .3f}")
    print(f"laion - (flipd / png length) = {df1['diff'].mean():.3f} +- {df1['diff'].std(): .3f}")
    print(f"sdxl - (flipd / png length) = {df2['diff'].mean():.3f} +- {df2['diff'].std(): .3f}")

    # 5) Build classification dataset (X=diff, Y=label)
    dataset = combined[['prompt','diff','label', 'value', 'flipd']].copy()
    dataset = dataset.dropna(subset=["diff", "label"])

    auc = roc_auc_score(dataset["label"], dataset["value"])
    print(f"ROC AUC (png length → file label): {auc:.4f}")
    auc = roc_auc_score(dataset["label"], dataset["diff"])
    print(f"ROC AUC (flipd / png length → file label): {auc:.4f}")
    print(f"Accuracy (png length → file label): {best_threshold_accuracy(dataset['value'], dataset['label']):.4f}")
    print(f"Accuracy (flipd / png length → file label): {best_threshold_accuracy(dataset['diff'], dataset['label']):.4f}")
    return
    print("Simple classifier flipd / png length")
    simple_binary_classifier(dataset["diff"], dataset["label"], max_depth=2)
    print("Simple classifier png length")
    simple_binary_classifier(dataset["value"], dataset["label"], max_depth=2)
    print("Simple classifier png flipd")
    simple_binary_classifier(dataset["flipd"], dataset["label"], max_depth=2)
    # Save to CSV
    # dataset.to_csv("diff_dataset.csv", index=False)
    # print("Saved classification dataset to diff_dataset.csv")


    # Separate the data by label
    values0 = dataset[dataset["label"] == 0]['flipd']
    values1 = dataset[dataset["label"] == 1]['flipd']
    # Plot histograms overlaid for the two label groups
    # plt.figure()
    # plt.hist(values0, bins=50, density=True, alpha=0.6, color='blue', label='Real Images')
    # plt.hist(values1, bins=50, density=True, alpha=0.6, color='orange', label='Generated Images')

    # # Labels and legend
    # plt.xlabel('PNG Length Ratio')
    # plt.ylabel('Density')
    # # plt.xlim(0, 1.5)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    def visualize_top_n(
        df: pd.DataFrame,
        json1_path: str,
        json2_path: str,
        images_dir1: str,
        images_dir2: str,
        n: int = 5,
        sort_ascending: bool = True,
        prompt_wrap_width: int = 20
    ):
        """
        Visualize top-n prompts with images from two sources and an "up/down" row.

        df: DataFrame with ['value','prompt','label']
        json1_path, json2_path: paths to JSON mappings {filename: prompt}
        images_dir1, images_dir2: dirs for JSON1 and JSON2 images
        n: number of unique top prompts to display
        sort_ascending: whether to sort df.value ascending
        prompt_wrap_width: max chars per line for prompt text
        """
        # 1) sort & select top-n unique prompts
        df_sorted = df.sort_values('flipd', ascending=sort_ascending)
        top_prompts = df_sorted['prompt'].drop_duplicates().iloc[:n].tolist()

        # 2) load mappings
        with open(json1_path, 'r', encoding='utf-8') as f:
            map1 = json.load(f)
        with open(json2_path, 'r', encoding='utf-8') as f:
            map2 = json.load(f)

        # 3) invert mapping: prompt -> [filenames]
        inv1 = {}
        for fn, pr in map1.items():
            inv1.setdefault(pr, []).append(fn)
        inv2 = {}
        for fn, pr in map2.items():
            inv2.setdefault(pr, []).append(fn)

        # 4) build figure with 3 rows: source1, source2, up/down
        fig, axes = plt.subplots(3, n, figsize=(4*n, 10))
        row_labels = ["Source 1", "Source 2", "Decision"]

        for col, prompt in enumerate(top_prompts):
            # wrap prompt
            wrapped = textwrap.fill(prompt, width=prompt_wrap_width)
            # label for this prompt
            lbl = df_sorted[df_sorted['prompt'] == prompt]['label'].iloc[0]
            # filenames
            fn1 = inv1.get(prompt, [None])[0]
            fn2 = inv2.get(prompt, [None])[0]

            # display source1 image
            ax1 = axes[0, col]
            ax1.axis('off')
            if col == 0:
                ax1.set_ylabel(row_labels[0], rotation=90, size='large')
            ax1.set_title(wrapped, fontsize=12, pad=10)
            if fn1 and os.path.isfile(os.path.join(images_dir1, fn1)):
                ax1.imshow(Image.open(os.path.join(images_dir1, fn1)))
            else:
                ax1.text(0.5, 0.5, "Missing", ha='center', va='center')

            # display source2 image
            ax2 = axes[1, col]
            ax2.axis('off')
            if col == 0:
                ax2.set_ylabel(row_labels[1], rotation=90, size='large')
            if fn2 and os.path.isfile(os.path.join(images_dir2, fn2)):
                ax2.imshow(Image.open(os.path.join(images_dir2, fn2)))
            else:
                ax2.text(0.5, 0.5, "Missing", ha='center', va='center')

            # display decision row: up/down
            ax3 = axes[2, col]
            ax3.axis('off')
            if col == 0:
                ax3.set_ylabel(row_labels[2], rotation=90, size='large')
            decision = 'up' if lbl == 1 else 'down'
            ax3.text(0.5, 0.5, decision, ha='center', va='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig("na_jutro/bottom_cond_lid_ai.png")
    
    # visualize_top_n(
    #     dataset, 
    #     json1_path="data/sdxl_samplesv2_ai_jpg/captions.json",
    #     json2_path="data/laion-2b-1024x1024/captions_ai.json",
    #     images_dir1="data/sdxl_samplesv2_ai_jpg/",
    #     images_dir2="data/laion-2b-1024x1024/",
    #     n=20,
    #     sort_ascending=True,
    # )

if __name__ == "__main__":
    main()


import re

def filter_prompts(df: pd.DataFrame, base_prompt: str, match_digits_only: bool = True) -> pd.DataFrame:
    if match_digits_only:
        # ^base_prompt(\d*)$  -> base_prompt optionally followed by digits only
        pattern = rf"^{re.escape(base_prompt)}\d*$"
        mask = df["prompt"].astype(str).str.match(pattern)
    else:
        mask = df["prompt"].astype(str).str.startswith(base_prompt)
    return df[mask].copy()


def plot_for_single_prompt():
    prompt = "a pair of white crocheted slippers are on a beige background"
    df = load_flipd_dataframe("diffusion_memorization/outputs/attributions/many-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json")
    df_filtered = filter_prompts(df, prompt)
    values = df_filtered["flipd"].dropna()
    print(f"LID: {values.mean()} +/- {values.std()}")

    plt.figure(figsize=(6,4))
    plt.hist(values, bins="auto")  # use automatic bins
    plt.title(f"Distribution of 'flipd' for prompts matching: {prompt}")
    plt.xlabel("flipd")
    plt.ylabel("count")
    plt.tight_layout()

    out_path = f'na_jutro/{prompt[:10].replace(" ", "_")}_conditional'
    plt.savefig(out_path)

# plot_for_single_prompt()
