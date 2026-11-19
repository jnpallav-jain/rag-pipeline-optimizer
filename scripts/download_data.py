"""Download the public data and model this project uses.

- SQuAD v1.1 dev set (Rajpurkar et al., 2016), CC BY-SA 4.0 -> data/raw/
- BAAI/bge-small-en-v1.5 embedding model (MIT license) -> the Hugging Face cache
"""
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = {
    "squad-dev-v1.1.json":
        "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
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

    sys.path.insert(0, str(ROOT))
    from rag_optimizer.pipeline import EMBED_MODEL, _encoder
    print(f"loading embedding model {EMBED_MODEL} (downloads once, ~130 MB) ...")
    _encoder()
    print("  ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
