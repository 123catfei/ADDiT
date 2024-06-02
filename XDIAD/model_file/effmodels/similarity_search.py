import faiss
import numpy as np
import torch

# Query-test feature  Memory--all train features
Query = torch.randn(196, 272).numpy().astype(np.float32)  # (196, 272)
Memory = torch.randn(1000, 272).numpy().astype(np.float32)  # (1000, 272)


class SimilaritySearch:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def reconstruct_vectors(self, memory, query_vectors, k=1):
        self.index.add(memory)  # 添加memory中的向量到索引
        _, indices = self.index.search(query_vectors, k)  # k=1表示找最相似的一个特征(nn search)
        return memory[indices.flatten()]


ss = SimilaritySearch(272)
res = ss.reconstruct_vectors(Memory, Query)
print(res.shape)  # (196, 272)

