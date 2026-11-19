"""Download the public data this project uses into data/raw/.

- SQuAD v1.1 dev set (Rajpurkar et al., 2016), CC BY-SA 4.0
- GloVe 50-d Wikipedia + Gigaword vectors (Pennington et al., 2014), via the gensim-data mirror
"""
import pathlib
import sys
import urllib.request

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "squad-dev-v1.1.json":
        "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
    "glove-wiki-gigaword-50.gz":
        "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
        "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz",
}


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"already present: {dest}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
