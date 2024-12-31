import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from .AITA_Agent import AITA_Agent

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

TEST_QUERIES = [
    """
    My brother and his now wife got married three days ago. A very small destination ceremony under 15 people total. My now fiancé and I extended our trip after everyone went home and spent a couple of days exploring the Grand Canyon, a couple hours north of the wedding, where he proposed.

    When I shared the news with my brother and now SIL, he responded with hostility saying that it looked like we were competing.

    I apologized, quickly realizing that he was advocating for my SIL and that she felt hurt (although I’m truly failing to understand why). I also texted her a separate apology and explained that it was not our intent to encroach and just wanted to share the news with family and that it’s my belief that there’s room for happiness for everyone. She did not respond.

    In response to my apology, my brother doubled down and said the timing and location were hurtful and that we shouldn’t planned around the wedding.
    """
]

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