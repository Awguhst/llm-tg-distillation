"""
Get the data.

1. The real Tg dataset (Kaggle, PolyInfo-derived) is already in the project folder.
   Move it into data/raw/ and print its shape and columns.
2. Download PI1M (about 1 million hypothetical polymer SMILES) from GitHub.
"""
import os
import shutil
import urllib.request

import pandas as pd

import config

os.makedirs(config.RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------- 1. real Tg dataset
filename = os.path.basename(config.REAL_TG_CSV)
if not os.path.exists(config.REAL_TG_CSV):
    # The file was downloaded by hand into the project root; move it into data/raw/.
    if os.path.exists(filename):
        shutil.move(filename, config.REAL_TG_CSV)
    else:
        raise SystemExit(f"Cannot find {filename} in the project folder or in {config.RAW_DIR}.")

real = pd.read_csv(config.REAL_TG_CSV)
print(f"Real Tg dataset: {config.REAL_TG_CSV}")
print(f"  rows: {len(real)}")
print(f"  columns: {list(real.columns)}")
print(real.head())

# ---------------------------------------------------------------- 2. PI1M
if os.path.exists(config.PI1M_CSV):
    print(f"\nPI1M already downloaded: {config.PI1M_CSV}")
else:
    downloaded = False
    for url in config.PI1M_URLS:
        print(f"\nDownloading PI1M from {url} ...")
        try:
            urllib.request.urlretrieve(url, config.PI1M_CSV)
            downloaded = True
            break
        except Exception as error:
            print(f"  failed: {error}")
    if not downloaded:
        raise SystemExit("Could not download PI1M. Check the URLs in config.py.")

pi1m = pd.read_csv(config.PI1M_CSV)
print(f"PI1M: {config.PI1M_CSV}")
print(f"  rows: {len(pi1m)}")
print(f"  columns: {list(pi1m.columns)}")
print(pi1m.head())
