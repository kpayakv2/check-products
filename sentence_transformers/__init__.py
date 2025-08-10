from . import util


class SentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, sentences, convert_to_tensor=True):
        import torch
        
        # สร้าง mock embeddings ที่มี shape ถูกต้อง
        # ใช้ embedding dimension = 384 (เหมือน MiniLM model)
        embedding_dim = 384
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # สร้าง random embeddings สำหรับแต่ละ sentence
        batch_size = len(sentences)
        embeddings = torch.randn(batch_size, embedding_dim)
        
        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.numpy()
