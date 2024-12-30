from llama_index.core.workflow import Workflow
from typing import List
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import json

async def evaluate_test_set(workflow: Workflow, df: pd.DataFrame) -> List:
  """
  Evaluate RAG Workflow using AITA test set
  """

  def create_complete_test_set(df):
    """
    Create a list of tuples that are a concatenation of the submission_text and submission_title, the top_comment_1, and the top_comment_1_classification
    """
    test_set = []
    for index, row in df.iterrows():
      test_set.append((row['submission_text'] + '\n\n' + row['submission_title'], row['top_comment_1'], row['top_comment_1_classification']))
    return test_set

  def create_small_test_set(df, samples_per_class=10):
    """
    Create a balanced test set with a specified number of samples per classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the columns 'submission_text', 'submission_title', 
        'top_comment_1', and 'top_comment_1_classification'
    samples_per_class : int, default=10
        Number of samples to include for each unique classification
        
    Returns:
    --------
    list of tuples
        Each tuple contains (concatenated submission text and title, top comment, classification)
    """
    # Create balanced sample
    balanced_df = (df.groupby('top_comment_1_classification')
                    .apply(lambda x: x.sample(n=min(len(x), samples_per_class)))
                    .reset_index(drop=True))
    
    # Create list of tuples
    test_set = []
    for _, row in balanced_df.iterrows():
        test_set.append((
            row['submission_text'] + '\n\n' + row['submission_title'],
            row['top_comment_1'],
            row['top_comment_1_classification']
        ))
    
    return test_set

  test_set = create_small_test_set(AITA_test_df, samples_per_class=50)

  results = []
  for i in tqdm(range(len(test_set)), desc="Evaluating test set"):
    try:
      # run the workflow with the test query
      result = await workflow.run(query=test_set[i][0])
      
      # collect response chunks
      response = ""
      async for chunk in result.async_response_gen():
        response += chunk  # Accumulate chunks
        #print(chunk, end="", flush=True)
      
      # store results
      temp = {
        'response': response,
        'query': test_set[i][0],
        'top_comment_1': test_set[i][1],
        'top_comment_1_classification': test_set[i][2]
      }
      results.append(temp)
      #pprint(temp)
      #print('\n')
        
    except Exception as e:
      print(f"Error processing sample {i}: {str(e)}")
      continue

  return results

# create the agent/workflow
workflow = AITA_RAG_Workflow(
  docs_to_retrieve=10,
  timeout=900
)

# run evaluation on test set
results = await evaluate_test_set(workflow, AITA_test_df)

with open('AITA_RAG_Agent_eval_results_second_run.json', 'w') as f:
    json.dump(results, f)