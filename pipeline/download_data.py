"""
Get the data.

1. The real Tg dataset: the curated collection of the Jablonka group from Zenodo
   (record, file name and md5 in config.py; CC-BY 4.0). The md5 is checked after the download.
2. PI1M (about 1 million hypothetical polymer SMILES) from GitHub.
"""
import hashlib
import os
import urllib.request

import pandas as pd

import config

os.makedirs(config.RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------- 1. real Tg dataset
if not os.path.exists(config.REAL_TG_CSV):
    print(f"Downloading {config.REAL_TG_FILE} (about 44 MB) from Zenodo record {config.REAL_TG_ZENODO_RECORD} ...")
    urllib.request.urlretrieve(config.REAL_TG_URL, config.REAL_TG_CSV)
md5 = hashlib.md5(open(config.REAL_TG_CSV, "rb").read()).hexdigest()
if md5 != config.REAL_TG_MD5:
    raise SystemExit(f"md5 mismatch for {config.REAL_TG_CSV}: {md5}, expected {config.REAL_TG_MD5}")
real = pd.read_csv(config.REAL_TG_CSV, usecols=list(config.REAL_COLUMNS.values()))
print(f"Real Tg dataset: {config.REAL_TG_CSV} (doi {config.REAL_TG_DOI}, md5 ok)")
print(f"  rows: {len(real)}")
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
