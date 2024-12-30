import os
from typing import List
from pprint import pprint

import pandas as pd

from pinecone import Pinecone, ServerlessSpec

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


class VectorStoreUtility:

    def replace_none_values(self, df):
        # Replace None in top_comment columns with 'No Comment'
        comment_cols = [col for col in df.columns if col.startswith('top_comment_') and not col.endswith('classification')]
        for col in comment_cols:
            df[col] = df[col].fillna('No Comment')
        
        # Replace None in classification columns with 'No Classification'
        classification_cols = [col for col in df.columns if col.endswith('classification')]
        for col in classification_cols:
            df[col] = df[col].fillna('No Classification')
        
        return df

    def convert_df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame rows to LlamaIndex Documents.
        Text will be concatenation of submission_title and submission_text.
        Top comment and its classification will be stored as metadata.

        Args:
            df (pd.DataFrame): DataFrame with columns submission_title, submission_text,
                            top_comment_1, and top_comment_1_classification

        Returns:
            List[Document]: List of LlamaIndex Document objects
        """
        documents = []

        for _, row in df.iterrows():
            # Combine title and text for document content
            text = f"Title: {row['submission_title']}\n\nContent: {row['submission_text']}\n\nCorrect Classification: {row['top_comment_1_classification']}\n\nCorrect Justification: {row['top_comment_1']}"

            # Create metadata from top comment and classification
            metadata = {}
            #for i in range(1, 4):  # top 3 comments and their classifications stored as metadata
            #    metadata[f'top_comment_{i}'] = row[f'top_comment_{i}']
            #    metadata[f'top_comment_{i}_classification'] = row[f'top_comment_{i}_classification']
            #metadata['submission_score'] = row['submission_score']
            #metadata['submission_date'] = row['submission_date']
            metadata['Submission URL'] = row['submission_url']
            #metadata['ambiguity_score'] = row['ambiguity_score']

            doc = Document(
                text=text,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def create_pinecone_vs_index(
        self,
        index_name: str,
        documents: List[Document],
        embed_model_provider: str = "openai",
        embed_model_endpoint: str = "text-embedding-3-small"
    ) -> VectorStoreIndex:
        # Create the vector store index and save it on pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        if embed_model_endpoint != "text-embedding-3-small":
            if embed_model_endpoint == "text-embedding-3-large":
                vs_index_dimensions = 3072
        else:
            vs_index_dimensions = 1536 # text-embedding-3-small is default

        pc.create_index(
            name=index_name,
            dimension=vs_index_dimensions, # matches openAI's text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        pinecone_index = pc.Index(index_name)

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if embed_model_provider != "openai":
            pass # add other embedding models here
        else: # use openai
            embed_model = OpenAIEmbedding(model=embed_model_endpoint, api_key=os.getenv('OPENAI_API_KEY'))

        parser = SentenceSplitter(chunk_size=8192)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embed_model=embed_model,
            node_parser=parser,
            show_progress=True
        )

        # print index description after inserting embeddings
        print(f'\nVS Index Description:\n')
        index_description = pc.describe_index(index_name)
        pprint(index_description)

        return index