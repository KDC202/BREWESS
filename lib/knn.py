import faiss
from .utils import check_numpy, free_memory


class NNS:
    def __init__(self, base, **kwargs):
        raise NotImplementedError()

    def search(self, query, k=1):
        raise NotImplementedError()


class ExactNNS(NNS):
    def __init__(self, base, device_id=0):
        assert len(base.shape) == 2
        dim = base.shape[1]
        # 创建了一个 Faiss 平坦索引（IndexFlatL2），该索引使用 L2 距离度量来计算向量之间的相似性
        self.index_flat = faiss.IndexFlatL2(dim)
        # 创建了一个 Faiss 标准 GPU 资源对象（StandardGpuResources），用于管理 GPU 资源
        self.res = faiss.StandardGpuResources()
        # 告诉 Faiss 在进行相似性搜索时不要使用临时内存
        self.res.noTempMemory()
        # 将一个在 CPU 上构建的 Faiss 索引对象转移到 GPU 上
        self.index_flat = faiss.index_cpu_to_gpu(self.res, device_id, self.index_flat)
        # 向 Faiss 索引中添加基准数据
        self.index_flat.add(check_numpy(base))

    def search(self, query, k=1):
        free_memory()
        assert len(query.shape) == 2
        _, neighbors = self.index_flat.search(check_numpy(query), k)
        free_memory()
        return neighbors
