import os
import pickle

def system_boot_bm25():
    print("bm25 boot...")
    base_dir = os.path.join(os.path.dirname(__file__), "local_files")
    file_path = os.path.join(base_dir, "precomputed_bm25.pkl")
    with open(file_path, "rb") as f:
        all_bm25s = pickle.load(f)
    """base_dir = os.path.join(os.path.dirname(__file__), "local_files")
    pickle_dir = os.path.join(base_dir, "bm25 pickle")
    txt_dir = os.path.join(base_dir, "Txt Files")
    file_names = [f for f in os.listdir(pickle_dir) if os.path.isfile(os.path.join(pickle_dir, f))]

    all_bm25s = []
    for file_name in file_names:
        pickle_path = os.path.join(pickle_dir, file_name)
        with open(pickle_path, "rb") as f:
            bm25 = pickle.load(f)
        name = file_name[5:-4]  # Strip prefix and '.pkl'
        print(name)
        txt_file_path = os.path.join(txt_dir, f"{name}_nodes.txt")
        with open(txt_file_path, "r", encoding="utf-8") as f:
            chunks = [line.strip() for line in f.readlines()]
        doc_vectors = {}
        for i, doc in enumerate(chunks):
            sparse_vec = bm25.encode_documents(doc)
            doc_vectors[i] = sparse_vec
        bm25_dict = {
            "name": name,
            "data": bm25,
            "vectors": doc_vectors
        }
        all_bm25s.append(bm25_dict)
"""
    return all_bm25s