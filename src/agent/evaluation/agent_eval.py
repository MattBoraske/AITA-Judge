import asyncio
import os
import argparse
import logging
from datetime import datetime
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
from multiprocessing import freeze_support

def setup_logging(log_level=logging.INFO, results_directory='eval_results'):
    """Set up logging configuration"""

    # create log directory
    os.makedirs(results_directory, exist_ok=True)
    log_filename = os.path.join(results_directory, f'agent_eval.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run AITA Agent evaluation with custom parameters')
    
    # Add logging level argument
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    
    # Workflow parameters
    parser.add_argument('--timeout', type=int, default=900,
                      help='Timeout in seconds (default: 900)')
    parser.add_argument('--llm-provider', type=str, default='openai',
                        help='LLM provider (default: openai)')
    parser.add_argument('--llm-endpoint', type=str, default='text-davinci-003',
                        help='LLM endpoint')
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-large',
                      help='Embedding model endpoint')
    parser.add_argument('--pinecone-index', type=str, default='aita-text-embedding-3-large',
                      help='Pinecone vector index name')
    parser.add_argument('--docs-to-retrieve', type=int, default=5,
                      help='Number of documents to retrieve')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, 
                      default='MattBoraske/reddit-AITA-submissions-and-comments-multiclass',
                      help='HuggingFace dataset path')
    parser.add_argument('--complete-dataset', action='store_true', default=False)
    parser.add_argument('--samples-per-class', type=int, default=1,
                      help='Number of samples per class')
    
    # Output file parameters
    parser.add_argument('--results-directory', type=str, default='eval_results')
    parser.add_argument('--responses-file', type=str, default='responses.json')
    parser.add_argument('--classification-report-filepath', type=str, 
                      default='classification_report.txt')
    parser.add_argument('--confusion-matrix-filepath', type=str, 
                      default='confusion_matrix.png')
    parser.add_argument('--mcc-filepath', type=str, default='mcc_score.json')
    parser.add_argument('--rouge-filepath', type=str, default='rouge_score.json')
    parser.add_argument('--bleu-filepath', type=str, default='bleu_score.json')
    parser.add_argument('--comet-filepath', type=str, default='comet_score.json')
    parser.add_argument('--toxicity-stats-filepath', type=str, 
                      default='toxicity_stats.json')
    parser.add_argument('--toxicity-plot-filepath', type=str, 
                      default='toxicity_plot.png')
    parser.add_argument('--retrieval-eval-filepath', type=str, 
                      default='retrieval_eval.json')
    parser.add_argument('--retrieval-eval-summary-filepath', type=str, 
                      default='retrieval_eval_summary.json')
    
    return parser.parse_args()

def run_evaluation(args, logger):
    """Run the evaluation process with logging"""
    logger.info("Starting evaluation process")
    logger.debug(f"Evaluation parameters: {vars(args)}")
    
    try:
        # Initialize evaluation utility
        logger.info("Initializing Evaluation Utility")
        eval_util = Evaluation_Utility()

        # Create results directory
        logger.debug(f"Creating results directory: {args.results_directory}")
        os.makedirs(args.results_directory, exist_ok=True)

        # Initialize workflow
        logger.info("Initializing AITA Agent workflow")
        workflow = AITA_Agent(
            timeout=args.timeout,
            llm_provider=args.llm_provider,
            llm_endpoint=args.llm_endpoint,
            embedding_model_endpoint=args.embedding_model,
            pinecone_vector_index=args.pinecone_index,
            docs_to_retrieve=args.docs_to_retrieve,
        )
        
        # Load and prepare dataset
        logger.info(f"Loading dataset from {args.dataset}")
        hf_dataset = load_dataset(args.dataset)
        AITA_test_df = hf_dataset['test'].to_pandas()
        
        logger.debug("Filtering out INFO classifications")
        AITA_test_df = AITA_test_df[AITA_test_df['top_comment_1_classification'] != 'INFO']
        logger.info(f"Dataset size after filtering: {len(AITA_test_df)}")
        
        # Create test set
        if not args.complete_dataset:
            logger.info(f"Creating balanced test set with {args.samples_per_class} samples per class")
            test_set = eval_util.create_balanced_test_set(
                AITA_test_df, 
                samples_per_class=args.samples_per_class
            )
        else:
            logger.info("Using complete dataset for testing")
            test_set = eval_util.create_test_set(AITA_test_df)
        
        logger.info(f"Final test set size: {len(test_set)}")
        
        # Collect responses
        logger.info("Starting response collection")
        responses = asyncio.run(eval_util.collect_responses(workflow, test_set))
        logger.info(f"Collected {len(responses)} responses")

        # Run evaluation
        logger.info("Starting evaluation of responses")
        eval_util.evaluate(
            responses=responses,
            results_directory=args.results_directory,
            classification_report_filepath=args.classification_report_filepath,
            confusion_matrix_filepath=args.confusion_matrix_filepath,
            mcc_filepath=args.mcc_filepath,
            rouge_filepath=args.rouge_filepath,
            bleu_filepath=args.bleu_filepath,
            comet_filepath=args.comet_filepath,
            toxicity_stats_filepath=args.toxicity_stats_filepath,
            toxicity_plot_filepath=args.toxicity_plot_filepath,
            retrieval_eval_filepath=args.retrieval_eval_filepath,
            retrieval_eval_summary_filepath=args.retrieval_eval_summary_filepath
        )

        # Save responses
        responses_path = os.path.join(args.results_directory, args.responses_file)
        logger.info(f"Saving responses to {responses_path}")
        with open(responses_path, 'w') as f:
            json.dump(responses, f)
            
        logger.info("Evaluation process completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation process: {str(e)}", exc_info=True)
        raise

def setup_telemetry(logger):
    """Set up telemetry with logging"""
    logger.info("Setting up telemetry")
    try:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
        logger.debug("Added Phoenix API key to environment")
        
        span_phoenix_processor = SimpleSpanProcessor(
            HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
        )
        
        tracer_provider = trace_sdk.TracerProvider()
        tracer_provider.add_span_processor(span_processor=span_phoenix_processor)
        
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Telemetry setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up telemetry: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Setup logging with specified level
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(log_level, args.results_directory)
    
    logger.info("Starting AITA Agent evaluation script")
    
    try:
        load_dotenv(find_dotenv())
        freeze_support()
        setup_telemetry(logger)
        run_evaluation(args, logger)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Script execution completed")