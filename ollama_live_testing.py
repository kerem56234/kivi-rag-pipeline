import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline.embedding import OllamaEmbeddingModel

embedder = OllamaEmbeddingModel()
vector = embedder.embed("where do i get good pizza?")
print(len(vector), vector[:5])

from rag_pipeline.generator import OllamaGeneratorModel

generator = OllamaGeneratorModel()
answer = generator.generate("hello world")
print(answer)
