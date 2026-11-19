import os

# FAISS and PyTorch each ship an OpenMP runtime; on macOS running both multi-threaded in one
# process segfaults. One thread also keeps latency measurements comparable across retrievers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
