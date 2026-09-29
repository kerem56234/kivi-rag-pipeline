from rag_pipeline.database import PsycopgDatabaseClient
from rag_pipeline.retriever import (
    PgVectorCandidateSource,
    PostgresLexicalCandidateSource,
    Retriever,
)
from rag_pipeline.embedding import OllamaEmbeddingModel
from rag_pipeline.generator import OllamaGeneratorModel
from rag_pipeline.integrator import RAGIntegrator

db = PsycopgDatabaseClient("postgresql://user:password@localhost:5432/your_db")

retriever = Retriever(
    vector_source=PgVectorCandidateSource(db),
    lexical_source=PostgresLexicalCandidateSource(db),
)

integrator = RAGIntegrator(
    embedding_model=OllamaEmbeddingModel(),
    retriever=retriever,
    generator_model=OllamaGeneratorModel(),
)

response = integrator.answer(
    user_prompt="What does pgai synchronize?",
    retrieval_view="wiki_embedding",
    result_limit=5,
)

print(response.answer)
print(response.sources)