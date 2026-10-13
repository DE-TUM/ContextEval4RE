"""Attention mass share and gold-vs-distractor selectivity per prompt type.

For every case folder under --data_dir (as produced by get_attention.py), and for each of the
four prompt types, this script computes from the last-layer attention (mean over generation steps):

  share_{pre,rel,demo,test}: fraction of total attention mass on each prompt region
  sel:   mean per-token attention on gold relation options / mean per-token attention on
         distractor options (1 = no preference, >1 = gold options are attended more)
  pctile: mean percentile rank of gold options among all options (0.5 = no preference)
  sel_local: like sel, but each gold option is compared only with nearby distractors
         (controls for position effects)

Gold options are the relations listed in the gt_relations prompt (P_ref) of the same case.
Usage:
  python attention_selectivity.py --data_dir attention_raw_results --out_csv attn_selectivity.csv
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

PROMPTS = {
    "P_base": "similar_example_3shot_prompts",
    "P_ref": "gt_relations_3shot_prompts",
    "P_full": "merged_relations_3shot_prompts",
    "P_rand": "random_example_3shot_prompts",
}
ENTRY = re.compile(r"([^\s,()][^(),]*?)\(([^()]*)\)")  # name(Type, Type)


def load(case_dir, prefix):
    attn = np.load(os.path.join(case_dir, prefix + "_attn.npy"))  # (gen_len, in_len)
    with open(os.path.join(case_dir, prefix + "_in_toks.json")) as f:
        toks = json.load(f)
    return attn, toks


def text_and_offsets(toks):
    parts = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in toks]
    ends = np.cumsum([len(p) for p in parts])
    return "".join(parts), np.stack([ends - [len(p) for p in parts], ends], axis=1)


def token_mask(offs, start, end):
    return (offs[:, 1] > start) & (offs[:, 0] < end)


def options(txt, start, end):
    return [(m.group(1).strip(), start + m.start(), start + m.end()) for m in ENTRY.finditer(txt[start:end])]


def regions(txt):
    o0 = txt.index("Ontology Relations:") + len("Ontology Relations:")
    e0 = txt.index("Example Sentence:")
    t0 = txt.index("Test Sentence:")
    return {"pre": (0, o0), "rel": (o0, e0), "demo": (e0, t0), "test": (t0, len(txt))}


def analyze_case(case_dir, window=5):
    attn, toks = load(case_dir, PROMPTS["P_ref"])
    txt, _ = text_and_offsets(toks)
    r = regions(txt)
    gold = {n for n, _, _ in options(txt, *r["rel"])}

    rows = []
    for name, prefix in PROMPTS.items():
        attn, toks = load(case_dir, prefix)
        txt, offs = text_and_offsets(toks)
        r = regions(txt)
        m = attn.mean(axis=0)
        row = {"prompt": name, "n_tokens": len(toks)}
        for k, (s, e) in r.items():
            row["share_" + k] = m[token_mask(offs, s, e)].sum() / m.sum()

        opts = options(txt, *r["rel"])
        vals = np.array([m[token_mask(offs, s, e)].mean() for _, s, e in opts])
        is_gold = np.array([n in gold for n, _, _ in opts])
        row["n_options"], row["n_gold"] = len(opts), int(is_gold.sum())
        if 0 < is_gold.sum() < len(is_gold):
            row["sel"] = vals[is_gold].mean() / vals[~is_gold].mean()
            ranks = vals.argsort().argsort() / (len(vals) - 1)
            row["pctile"] = ranks[is_gold].mean()
            w = window if name != "P_full" else 3 * window
            local = []
            for i in np.where(is_gold)[0]:
                nb = [j for j in range(max(0, i - w), min(len(vals), i + w + 1)) if not is_gold[j]]
                if nb:
                    local.append(vals[i] / vals[nb].mean())
            row["sel_local"] = float(np.mean(local)) if local else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_csv", default="attn_selectivity.csv")
    args = ap.parse_args()

    rows = []
    for case in sorted(os.listdir(args.data_dir)):
        case_dir = os.path.join(args.data_dir, case)
        if not os.path.isdir(case_dir):
            continue
        for row in analyze_case(case_dir):
            row["case"] = case
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    g = df.groupby("prompt", sort=False)
    summary = pd.DataFrame({
        "Rel(%)": g["share_rel"].mean() * 100,
        "Demo(%)": g["share_demo"].mean() * 100,
        "Test(%)": g["share_test"].mean() * 100,
        "Sel": g["sel"].mean(),
        "Sel_median": g["sel"].median(),
        "Sel>1": g["sel"].apply(lambda s: f"{int((s > 1).sum())}/{int(s.notna().sum())}"),
        "Pctile": g["pctile"].mean(),
        "Sel_local": g["sel_local"].mean(),
        "n_options": g["n_options"].mean(),
    })
    pd.set_option("display.width", 200)
    print(summary.round(2))
    print(f"\n{df['case'].nunique()} cases; per-case results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
