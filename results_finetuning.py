import json
import numpy as np
import pandas as pd


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


path = "diffusion_memorization/outputs/attributions/dog-original-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json"
# path = "diffusion_memorization/outputs/attributions/dog-lora-no-prior-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json"
# path = "diffusion_memorization/outputs/attributions/dog-lora-prior-lid-estimates-sdxl-conditional-ai-prompts-0.05_flipd.json"

prompts = json.load(open("diffusion_memorization/custom_captions.json", "r", encoding="utf-8"))

def main():
    df = load_flipd_dataframe(path)
    df = df.dropna()
    df['base_prompt'] = df['prompt'].str.replace(r'\s*\d+\s*$', '', regex=True)
    df = df[df["base_prompt"].isin(prompts)]
    print(df["flipd"].mean(), len(df))


if __name__ == "__main__":
    main()
