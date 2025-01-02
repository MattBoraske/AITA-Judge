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
from multiprocessing import freeze_support


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
    
    # Output file parameters
    parser.add_argument('--results-directory', type=str, default='eval_results',
                        help='Output directory for evaluation results (default: eval_results)')
    parser.add_argument('--responses-file', type=str, default='responses.json',
                        help='Output JSON file path for responses (default: responses.json)')
    parser.add_argument('--classification-report-filepath', type=str, default='classification_report.txt',
                        help='Output JSON file path for classification report (default: classification_report.txt)')
    parser.add_argument('--confusion-matrix-filepath', type=str, default='confusion_matrix.png',
                        help='Output PNG file path for confusion matrix (default: confusion_matrix.png)')
    parser.add_argument('--mcc-filepath', type=str, default='mcc_score.json',
                        help='Output file path for Matthews Correlation Coefficent (MCC) (default: mcc_score.json)')
    parser.add_argument('--rouge-filepath', type=str, default='rouge_score.json',
                        help='Output file path for ROUGE score (default: rouge_score.json)')
    parser.add_argument('--bleu-filepath', type=str, default='bleu_score.json',
                        help='Output file path for BLEU score (default: bleu_score.json)')
    parser.add_argument('--comet-filepath', type=str, default='comet_score.json',
                        help='Output file path for COMET score (default: comet_score.json)')
    parser.add_argument('--toxicity-stats-filepath', type=str, default='toxicity_stats.json',
                        help='Output file path for toxicity statistics (default: toxicity_stats.json)')
    parser.add_argument('--toxicity-plot-filepath', type=str, default='toxicity_plot.png',
                        help='Output file path for toxicity plot (default: toxicity_plot.png)')
    parser.add_argument('--retrieval-eval-filepath', type=str, default='retrieval_eval.json',
                        help='Output file path for retrieval evaluation (default: retrieval_eval.json)')
    parser.add_argument('--retrieval-eval-summary-filepath', type=str, default='retrieval_eval_summary.json',
                        help='Output file path for retrieval evaluation summary (default: retrieval_eval_summary.json)')
    
    return parser.parse_args()

def run_evaluation(args):
    # get eval util
    eval_util = Evaluation_Utility()

    # create results directory
    os.makedirs(args.results_directory, exist_ok=True)

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
        test_set = eval_util.create_balanced_test_set(
            AITA_test_df, 
            samples_per_class=args.samples_per_class
        )
    else:
        test_set = eval_util.create_test_set(AITA_test_df)
    
    # collect responses
    responses = asyncio.run(eval_util.collect_responses(workflow, test_set))

    # evaluate responses
    eval_util.evaluate(
        responses,
        args.results_directory,
        args.classification_report_filepath,
        args.confusion_matrix_filepath,
        args.mcc_filepath,
        args.rouge_filepath,
        args.bleu_filepath,
        args.comet_filepath,
        args.toxicity_stats_filepath,
        args.toxicity_plot_filepath,
        args.retrieval_eval_filepath,
        args.retrieval_eval_summary_filepath
    )

    # save responses
    with open(os.path.join(args.results_directory, args.responses_file), 'w') as f:
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
    freeze_support
    setup_telemetry()
    args = parse_args()
    run_evaluation(args)