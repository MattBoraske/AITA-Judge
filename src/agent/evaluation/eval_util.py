import pandas as pd
from llama_index.core.workflow import Workflow
from tqdm import tqdm


class Evaluation_Utility():

    def create_test_set(df: pd.DataFrame) -> list[tuple]:
        """
        Create a list of tuples that are a concatenation of the submission_text and submission_title, the top_comment_1, and the top_comment_1_classification
        """

        test_set = []
        for index, row in df.iterrows():
            test_set.append((row['submission_text'] + '\n\n' + row['submission_title'], row['top_comment_1'], row['top_comment_1_classification']))
        return test_set    

    
    def create_balanced_test_set(df: pd.DataFrame, samples_per_class: int = 10) -> list[tuple]:
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

    async def collect_responses(workflow: Workflow, test_set: list[tuple]):
        """
        Collect responses from the workflow for each sample in the test set.
        """

        test_responses = []

        for i in tqdm(range(len(test_set)), desc='Generating agent responses on test set.'):
            try:
                result = await workflow.run(query=test_set[i][0])

                # get retrieved doc contents
                retrieved_doc_contents = []
                for node in result.source_nodes:
                    retrieved_doc_contents.append(node.text)
                # get response
                response = ""
                async for chunk in result.async_response_gen():
                    response += chunk

                test_responses.append({
                    'response': response,
                    'query': test_set[i][0],
                    'retrieved_docs': retrieved_doc_contents,
                    'top_comment_1': test_set[i][1],
                    'top_comment_1_classification': test_set[i][2]
                })
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        return test_responses

    def evaluate():
        pass
    
