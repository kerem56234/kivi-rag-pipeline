import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from rag_pipeline.database import PsycopgDatabaseClient
from rag_pipeline.retriever import (
    PgVectorCandidateSource,
    PostgresLexicalCandidateSource,
    Retriever,
)
from rag_pipeline.embedding import OllamaEmbeddingModel
from rag_pipeline.generator import OllamaGeneratorModel
from rag_pipeline.integrator import RAGIntegrator

DSN  = "postgresql://user:password@localhost:5432/your_db"

with PsycopgDatabaseClient(DSN) as db:

        retriever = Retriever(
                vector_source=PgVectorCandidateSource(db),
                lexical_source=PostgresLexicalCandidateSource(db),
        )

        integrator = RAGIntegrator(
                embedding_model=OllamaEmbeddingModel(),
                retriever=retriever,
                generator_model=OllamaGeneratorModel(),
        )

        frage = input("stelle eine Frage: ")

        response = integrator.answer(
                user_prompt= frage,
                retrieval_view="public.delivery_order_items_embeddings2",
                result_limit=5,
        )

        print(response.answer)
        print(response.sources)
