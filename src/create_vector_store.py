from datasets import load_dataset

AITA_dataset = load_dataset('MattBoraske/reddit-AITA-submissions-and-comments-multiclass')
AITA_train_df = AITA_dataset['train'].to_pandas()
AITA_test_df = AITA_dataset['test'].to_pandas()

# get rid of INFO classification
AITA_train_df = AITA_train_df[AITA_train_df['top_comment_1_classification'] != 'INFO']
AITA_test_df = AITA_test_df[AITA_test_df['top_comment_1_classification'] != 'INFO']

from llama_index.core import Document
import pandas as pd
from typing import List

def replace_none_values(df):
    # Replace None in top_comment columns with 'No Comment'
    comment_cols = [col for col in df.columns if col.startswith('top_comment_') and not col.endswith('classification')]
    for col in comment_cols:
        df[col] = df[col].fillna('No Comment')
    
    # Replace None in classification columns with 'No Classification'
    classification_cols = [col for col in df.columns if col.endswith('classification')]
    for col in classification_cols:
        df[col] = df[col].fillna('No Classification')
    
    return df

def convert_df_to_documents(df: pd.DataFrame) -> List[Document]:
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

#AITA_df = replace_none_values(AITA_df)
AITA_documents = convert_df_to_documents(AITA_train_df)

from google.colab import userdata
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter

def create_pinecone_vs_index(index_name: str) -> VectorStoreIndex:
  # Create the vector store index and save it on pinecone
  pc = Pinecone(api_key=userdata.get('PINECONE_API_KEY'))

  pc.create_index(
      name=index_name,
      dimension=1536, # matches openAI's text-embedding-3-small
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1"),
  )

  pinecone_index = pc.Index(index_name)

  vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=userdata.get('OPENAI_API_KEY'))

  parser = SentenceSplitter(chunk_size=8192)

  index = VectorStoreIndex.from_documents(
      documents=AITA_documents,
      storage_context=storage_context,
      embed_model=embed_model,
      node_parser=parser,
      show_progress=True
  )

  return index

index = create_pinecone_vs_index('aita-text-embedding-3-small')