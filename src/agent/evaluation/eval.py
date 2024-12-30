import asyncio
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from datasets import load_dataset
import json
from ..AITA_Agent import AITA_Agent
from .eval_util import Evaluation_Utility
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

def parse_args():
    parser = argparse.ArgumentParser(description='Run AITA Agent evaluation with custom parameters')
    
    # Workflow parameters
    parser.add_argument('--timeout', type=int, default=900,
                      help='Timeout in seconds (default: 900)')
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-large',
                      help='Embedding model endpoint (default: text-embedding-3-large)')
    parser.add_argument('--pinecone-index', type=str, default='aita-text-embedding-3-large',
                      help='Pinecone vector index name')
    parser.add_argument('--docs-to-retrieve', type=int, default=5,
                      help='Number of documents to retrieve (default: 5)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MattBoraske/reddit-AITA-submissions-and-comments-multiclass',
                      help='HuggingFace dataset path')
    parser.add_argument('--complete-dataset', action='store_true', default=False)
    parser.add_argument('--samples-per-class', type=int, default=1,
                      help='Number of samples per class for evaluation (default: 1)')
    parser.add_argument('--output-file', type=str, default='AITA_RAG_Agent_eval_results.json',
                      help='Output file path for results (default: AITA_RAG_Agent_eval_results.json)')
    
    return parser.parse_args()

def run_evaluation(args):
    # get the workflow
    workflow = AITA_Agent(
        timeout=args.timeout,
        embedding_model_endpoint=args.embedding_model,
        pinecone_vector_index=args.pinecone_index,
        docs_to_retrieve=args.docs_to_retrieve,
    )
    
    # get the test data
    hf_dataset = load_dataset(args.dataset)
    AITA_test_df = hf_dataset['test'].to_pandas()
    # get rid of INFO classification
    AITA_test_df = AITA_test_df[AITA_test_df['top_comment_1_classification'] != 'INFO']
    
    if not args.complete_dataset:
        # create the test set
        test_set = Evaluation_Utility.create_balanced_test_set(
            AITA_test_df, 
            samples_per_class=args.samples_per_class
        )
    else:
        test_set = Evaluation_Utility.create_test_set(AITA_test_df)
    
    # collect responses
    responses = asyncio.run(Evaluation_Utility.collect_responses(workflow, test_set))
    
    # save responses
    print(f'Saving responses to {args.output_file}')
    with open(args.output_file, 'w') as f:
        json.dump(responses, f)

def setup_telemetry():
    # Add Phoenix API Key for tracing
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
    
    # Add Phoenix
    span_phoenix_processor = SimpleSpanProcessor(
        HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
    )
    
    # Add them to the tracer
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(span_processor=span_phoenix_processor)
    
    # Instrument the application
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    args = parse_args()
    setup_telemetry()
    run_evaluation(args)