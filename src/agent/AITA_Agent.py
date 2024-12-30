import os

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.groq import Groq

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.postprocessor.cohere_rerank import CohereRerank

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

from .AITA_prompts import AITA_Prompt_Library

class RetrieverEvent(Event):

  nodes: list[NodeWithScore]

class RerankEvent(Event):

  nodes: list[NodeWithScore]


class AITA_Agent(Workflow):

  def __init__(
        self,
        timeout=60,
        llm_provider='openai',
        llm_endpoint='gpt-4o-mini',
        embedding_model_endpoint='text-embedding-3-small',
        pinecone_vector_index='aita-text-embedding-3-small',
        docs_to_retrieve=3
    ):
        super().__init__(timeout=timeout)
        self.LLM_PROVIDER = llm_provider
        self.LLM_ENDPOINT  = llm_endpoint
        self.EMBEDDING_MODEL_ENDPOINT = embedding_model_endpoint
        self.PINECONE_VECTOR_INDEX  = pinecone_vector_index
        self.DOCS_TO_RETRIEVE  = docs_to_retrieve
        self.prompts = AITA_Prompt_Library.PROMPTS

  @step
  async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
    """Entry point for RAG"""
    #print('BEGIN RETRIEVE EVENT')

    # query parameter triggers the event
    query = ev.get('query')

    if not query:
      print('ERROR: No query provided.')
      return None

    # store query in global context
    await ctx.set("query", query)

    # get embedding model
    embed_model = OpenAIEmbedding(model=self.EMBEDDING_MODEL_ENDPOINT, api_key=os.getenv('OPENAI_API_KEY'))

    # get pinecone vectorstore
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    pinecone_index = pc.Index(self.PINECONE_VECTOR_INDEX)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create the index
    index = VectorStoreIndex.from_vector_store(
      vector_store=vector_store,
      storage_context=storage_context,
      embed_model=embed_model
    )

    # retrieve nodes
    retriever = index.as_retriever(similarity_top_k=self.DOCS_TO_RETRIEVE)
    nodes = await retriever.aretrieve(query)

    return RetrieverEvent(nodes=nodes)

  @step
  async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
    """Rerank retrieved nodes"""
    #print('BEGIN RERANK EVENT')

    cohere_reranker = CohereRerank(api_key=os.getenv('COHERE_API_KEY'), top_n=self.DOCS_TO_RETRIEVE)

    new_nodes = cohere_reranker.postprocess_nodes(
        ev.nodes, query_str=await ctx.get("query", default=None)
    )

    return RerankEvent(nodes=new_nodes)

  @step
  async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
    """Return a streaming response using reranked nodes"""
    #print('BEGIN SYNTHESIZE EVENT')

    if self.LLM_PROVIDER == 'groq':
      llm = Groq(model=self.LLM_ENDPOINT, api_key=os.getenv('GROQ_API_KEY'))
    else: # openai is default
      llm = OpenAI(model=self.LLM_ENDPOINT, api_key=os.getenv('OPENAI_API_KEY'))

    response_synthesizer = get_response_synthesizer(
      response_mode="refine",
      llm=llm,
      text_qa_template=self.prompts['AITA_text_qa_template'],
      refine_template=self.prompts['AITA_refine_qa_template'],
      streaming=True,
      verbose=True
    )

    query = await ctx.get("query", default=None)
    response = await response_synthesizer.asynthesize(query, nodes=ev.nodes)

    return StopEvent(result=response)