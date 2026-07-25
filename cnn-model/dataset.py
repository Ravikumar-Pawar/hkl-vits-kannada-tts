"""
CNN-Based Kannada TTS
Dataset Pipeline

Reuses the same OpenSLR Kannada dataset (kn_in_male) as HKL-VITS.

cnn-model/dataset.py
"""

import os
import zipfile

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================
# Downloader  (same URLs as HKL-VITS)
# ============================================

class DatasetDownloader:

    def __init__(self, config):
        self.config = config
        os.makedirs(config.DATA_DIR, exist_ok=True)

    def _download(self, url, path):
        print("Downloading:", url)
        r = requests.get(url, stream=True, timeout=60)
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            for chunk in tqdm(r.iter_content(1024), total=total // 1024, unit="KB"):
                f.write(chunk)

    def run(self):
        zip_path = os.path.join(self.config.DATA_DIR, self.config.ZIP_NAME)
        tsv_path = os.path.join(self.config.DATA_DIR, self.config.TSV_NAME)

        if not os.path.exists(zip_path):
            self._download(self.config.WAV_ZIP_URL, zip_path)
        else:
            print("ZIP exists — skipping")

        if not os.path.exists(tsv_path):
            self._download(self.config.TSV_URL, tsv_path)
        else:
            print("TSV exists — skipping")

        if not os.listdir(self.config.WAV_DIR) if os.path.exists(self.config.WAV_DIR) else True:
            os.makedirs(self.config.WAV_DIR, exist_ok=True)
            print("Extracting WAVs...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.config.WAV_DIR)
            print("Extraction complete")

        if not os.path.exists(self.config.TEXT_DIR):
            os.makedirs(self.config.TEXT_DIR, exist_ok=True)
            df = pd.read_csv(tsv_path, sep="\t", header=None, names=["wav_file", "text"])
            for _, row in df.iterrows():
                txt_path = os.path.join(self.config.TEXT_DIR, str(row["wav_file"]).strip() + ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(str(row["text"]).strip())
            print("Text files prepared:", len(df))


# ============================================
# PyTorch Dataset
# ============================================

class KannadaCNNDataset(Dataset):

    def __init__(self, text_dir, wav_dir, tokenizer, max_len=128):
        self.items     = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for fname in os.listdir(text_dir):
            if not fname.endswith(".txt"):
                continue
            wav_path = os.path.join(wav_dir, fname.replace(".txt", ".wav"))
            if not os.path.exists(wav_path):
                continue
            with open(os.path.join(text_dir, fname), encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                self.items.append({"text": text, "wav": wav_path})

        print("CNN Dataset size:", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text = self.items[idx]["text"]
        enc  = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
        }


# ============================================
# Collator
# ============================================

def cnn_collate(batch):
    return {
        "input_ids":      torch.stack([x["input_ids"]      for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }
