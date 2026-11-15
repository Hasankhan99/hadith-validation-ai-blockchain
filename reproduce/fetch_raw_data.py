"""
Clones the three public source datasets into data/ at the exact commits used in the thesis.
The processed splits in data/processed_v2 and data/processed_3class are already in the repository;
the raw data is only needed to rebuild them (src_v2/dataset_loader.py, src_v2/dataset_loader_3class.py)
or to run the V3 sanad supplement.

  python reproduce/fetch_raw_data.py
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {
    "MAHADDAT": ("https://github.com/kamelgaanoun/mhdetection.git", "baa237a59fe174ad20ff8be2df2cb15e9eeb3d6d"),
    "LK-Hadith-Corpus": ("https://github.com/ShathaTm/LK-Hadith-Corpus.git", "c45fa64aa40bf1d4a34b9b35359640a583b02881"),
    "AR-Sanad-280K": ("https://github.com/somaia02/Narrator-Disambiguation.git", "ddde1960dd4af1e50ecbe36ff5b92dee0c48bcf4"),
}

def run(*cmd):
    print(">", " ".join(cmd)); subprocess.run(cmd, check=True)

for name, (url, commit) in SOURCES.items():
    dst = os.path.join(ROOT, "data", name)
    if not os.path.isdir(os.path.join(dst, ".git")):
        run("git", "clone", url, dst)
    run("git", "-C", dst, "checkout", "--quiet", commit)
    head = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{name}: {head} {'OK' if head == commit else 'MISMATCH'}")
    if head != commit:
        sys.exit(1)
print("Raw datasets ready in data/.")
