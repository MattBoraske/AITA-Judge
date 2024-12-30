from datasets import load_dataset
from vs_util import VectorStoreUtility
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import os
import argparse

# create VectorStoreUtility object
vs_util = VectorStoreUtility()

def create_pinecone_vs(
        dataset_df: pd.DataFrame,
        index_name: str,
        embed_model_provider: str,
        embed_model_endpoint: str
    ):
    """
    Main function to create a Pinecone vector store from the AITA dataset loaded from HuggingFace.
    """

    try:
        print(f'\nCreating Pinecone Vector Store: {index_name}\n')

        # convert dataframe to list of LlamaIndex Document objects
        AITA_documents = vs_util.convert_df_to_documents(dataset_df)
        print(f'Created {len(AITA_documents)} LlamaIndex documents from dataset.\n')

        # create Pinecone vector store index
        vs_index = vs_util.create_pinecone_vs_index(
            index_name=index_name,
            documents=AITA_documents,
            embed_model_provider=embed_model_provider,
            embed_model_endpoint=embed_model_endpoint,
        )

        # print desciprtion of the Pinecone vector store index
        print(f'\nPinecone Vector Store Successfully Created: {index_name}\n')

    except Exception as e:
        print(f'Error creating Pinecone Vector Store: {str(e)}')

def create_dataset(hf_dataset_name: str) -> pd.DataFrame:
    # load training partition of AITA dataset from huggingface as pandas dataframe
    dataset = load_dataset(hf_dataset_name)
    df = dataset['train'].to_pandas()

    # get rid of INFO classification
    df = df[df['top_comment_1_classification'] != 'INFO']

    # replace None values in dataframe
    df = vs_util.replace_none_values(df)

    return df

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a Pinecone vector store from AITA dataset.')
    
    parser.add_argument('--dataset', 
                      type=str,
                      default='MattBoraske/reddit-AITA-submissions-and-comments-multiclass',
                      help='HuggingFace dataset name')
    
    parser.add_argument('--index-name',
                      type=str,
                      default='aita-text-embedding-3-large',
                      help='Name of the Pinecone vector store index')
    
    parser.add_argument('--embed-provider',
                      type=str,
                      default='openai',
                      help='Embedding model provider')
    
    parser.add_argument('--embed-endpoint',
                      type=str,
                      default='text-embedding-3-large',
                      help='Embedding model endpoint')
    
    return parser.parse_args()

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    # Parse command line arguments
    args = parse_arguments()

    dataset_df = create_dataset(args.dataset)

    create_pinecone_vs(
        dataset_df, 
        args.index_name, 
        args.embed_provider, 
        args.embed_endpoint
    )