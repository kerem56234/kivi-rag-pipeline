Retrieval Augmented Generation Pipeline for small scale experimental use at kivitendo gmbh.
Goals are to implement a local chatbot accessing customer data stored in postgres databases efficiently and generating
answers based on such data.

# Technologies
We are using postgres and psql as our database software. We are using the pgai extension on postgres for embedding
customer data. We are using the pgvector extension for postgres to store the Embeddings and perform similarity search 
on them. We are using python for scripting.

# Current State
Embeddings can be created automatically using pgai. The storing of the embeddings has been worked out with pgvector.
The current goal is to implement an efficient and effective retrieval system and a useful Chatbot interface.

Für weitere Dokumentationen, kontaktieren Sie kerem@kivitendo.de.
