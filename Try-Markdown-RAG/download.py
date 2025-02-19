import torch
from transformers import AutoModel
from numpy.linalg import norm
import pandas as pd

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True, torch_dtype=torch.bfloat16)

# 读取CSV文件
df = pd.read_csv('database_chunks_gemini.csv')

# 计算查询文本的嵌入
query = "推力公式是什么"
query_embedding = model.encode([query])[0]
print(query_embedding.shape)
# 遍历每个文本块并计算相似度
for idx, row in df.iterrows():
    text = row['文本内容']
    text_embedding = model.encode([text])[0]
    similarity = cos_sim(query_embedding, text_embedding)
    print(f"块 {idx+1} 相似度: {similarity:.4f}")
    print(f"文本内容: {text}\n")