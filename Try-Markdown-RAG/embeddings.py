from transformers import AutoModel
from langchain.embeddings.base import Embeddings
from transformers import AutoModel
import torch

# 创建一个自定义的嵌入类
class CustomJinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-zh"):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.eval()  # 设置为评估模式

    def embed_documents(self, texts):
        # 批量处理文档
        with torch.no_grad():
            embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        # 处理单个查询
        with torch.no_grad():
            embedding = self.model.encode([text])[0]
        return embedding.tolist()
