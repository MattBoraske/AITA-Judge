import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from agent.AITA_Agent import AITA_Agent

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from agent.test_queries import TEST_QUERIES

# define the workflow to run
AITA_workflow = AITA_Agent(
    docs_to_retrieve=5,
    timeout=300
)

async def single_response_CLI():
    async with asyncio.timeout(600):  # 10 minutes timeout
        # test query
        query = TEST_QUERIES[0]

        # run the workflow
        print('\nQUERY')
        print('-'*10)
        print(query)
        print('-'*10)
        print('\nRESPONSE') 
        print('-'*10)
        result = await AITA_workflow.run(query=query)
        async for chunk in result.async_response_gen():
            print(chunk, end="", flush=True)
        print()
        print('-'*10)

# Make this script runnable from the shell to test the workflow execution
if __name__ == "__main__":
    load_dotenv(find_dotenv())

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

    # Run workflow
    asyncio.run(single_response_CLI())