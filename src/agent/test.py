from AITA_prompts import AITA_Prompt_Library

from llama_index.core import PromptTemplate

text_qa_template = AITA_Prompt_Library.PROMPTS['AITA_text_qa_template']


prompt = text_qa_template.format(
    query_str="QUERY STRING"
)

print(prompt)