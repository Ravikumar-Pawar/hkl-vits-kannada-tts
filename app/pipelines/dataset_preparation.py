"""
HKL-VITS Hybrid Kannada TTS

Dataset Preparation Pipeline

Downloads:
    Kannada OpenSLR dataset

Prepares:
    wav files
    text files

Compatible with:
    Training pipeline


app/pipelines/dataset_preparation.py
"""

import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm
# ============================================
# CONFIG
# ============================================


class DatasetConfig:

    WAV_ZIP_URL = "https://openslr.trmal.net/resources/79/kn_in_male.zip"

    TSV_URL = "https://openslr.trmal.net/resources/79/line_index_male.tsv"

    DATA_DIR = os.path.join("dataset", "kannada_tts")

    WAV_DIR = os.path.join(DATA_DIR, "wavs")

    TEXT_DIR = os.path.join(DATA_DIR, "text")

    ZIP_NAME = "kannada_wavs.zip"

    TSV_NAME = "metadata.tsv"


# ============================================
# DOWNLOADER
# ============================================


class DatasetDownloader:

    def __init__(self, config):

        self.config = config

        os.makedirs(config.DATA_DIR, exist_ok=True)

        self.zip_path = os.path.join(config.DATA_DIR, config.ZIP_NAME)

        self.tsv_path = os.path.join(config.DATA_DIR, config.TSV_NAME)

    def download_file(self, url, output):

        print("\nDownloading:", url)

        response = requests.get(url, stream=True, timeout=60)

        total = int(response.headers.get("content-length", 0))

        with open(output, "wb") as file:

            for chunk in tqdm(
                response.iter_content(chunk_size=1024), total=total // 1024, unit="KB"
            ):

                file.write(chunk)

        print("Saved:", output)

    def run(self):

        if not os.path.exists(self.zip_path):

            self.download_file(self.config.WAV_ZIP_URL, self.zip_path)

        else:

            print("ZIP exists")

        if not os.path.exists(self.tsv_path):

            self.download_file(self.config.TSV_URL, self.tsv_path)

        else:

            print("TSV exists")


# ============================================
# EXTRACTOR
# ============================================


class DatasetExtractor:

    def __init__(self, config):

        self.config = config

        os.makedirs(config.WAV_DIR, exist_ok=True)

    def run(self):

        zip_path = os.path.join(self.config.DATA_DIR, self.config.ZIP_NAME)

        print("Extracting WAV files...")

        with zipfile.ZipFile(zip_path, "r") as zip_file:

            zip_file.extractall(self.config.WAV_DIR)

        print("Extraction completed")


# ============================================
# TEXT PREPARATION
# ============================================


class TextPreparer:

    def __init__(self, config):

        self.config = config

    def run(self):

        tsv_path = os.path.join(self.config.DATA_DIR, self.config.TSV_NAME)

        print("Preparing text files...")

        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["wav_file", "text"])

        os.makedirs(self.config.TEXT_DIR, exist_ok=True)

        count = 0

        for _, row in df.iterrows():

            wav_name = str(row["wav_file"]).strip()

            text = str(row["text"]).strip()

            txt_path = os.path.join(self.config.TEXT_DIR, wav_name + ".txt")

            with open(txt_path, "w", encoding="utf-8") as f:

                f.write(text)

            count += 1

        print("Text files:", count)

        return count


# ============================================
# MAIN PIPELINE
# ============================================


class KannadaTTSDatasetPipeline:

    def __init__(self):

        self.config = DatasetConfig()

        self.downloader = DatasetDownloader(self.config)

        self.extractor = DatasetExtractor(self.config)

        self.text = TextPreparer(self.config)

    def run(self):

        print()

        print("=" * 60)

        print("HKL-VITS DATASET PREPARATION")

        print("=" * 60)

        print("Dataset:", self.config.DATA_DIR)

        self.downloader.run()

        self.extractor.run()

        total = self.text.run()

        print()

        print("=" * 60)

        print("DATASET READY")

        print("Samples:", total)

        print("=" * 60)


# ============================================
# Standalone execution
# ============================================


if __name__ == "__main__":

    pipeline = KannadaTTSDatasetPipeline()

    pipeline.run()
