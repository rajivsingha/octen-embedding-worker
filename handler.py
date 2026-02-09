import runpod
import torch
from sentence_transformers import SentenceTransformer

print("Loading Octen-Embedding-8B model...")
model = SentenceTransformer(
    "Octen/Octen-Embedding-8B",
    model_kwargs={"torch_dtype": torch.float16},
    trust_remote_code=True
)
print(f"Model loaded! Embedding dim: {model.get_sentence_embedding_dimension()}")


def handler(job):
    try:
        input_data = job["input"]
        texts = input_data.get("texts", [])

        if not texts:
            return {"error": "No texts provided"}

        embeddings = model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})