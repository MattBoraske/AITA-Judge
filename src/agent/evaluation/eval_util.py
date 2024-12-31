import os
import re
import json
import pandas as pd
from llama_index.core.workflow import Workflow
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate
from transformers import pipeline
import torch

class Evaluation_Utility():
    """
    A utility class for evaluating AITA (Am I The Asshole) classifications and justifications.
    
    This class provides methods for creating test sets, collecting responses from a workflow,
    and evaluating both classifications and justifications using various metrics including
    ROUGE, BLEU, COMET scores, and classification metrics.
    """

    AITA_classifications = ['NTA', 'YTA', 'NAH', 'ESH']

    def create_test_set(self, df: pd.DataFrame) -> list[dict]:
        """
        Create a test set from a DataFrame containing submission text, titles, and comments.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns 'submission_text', 'submission_title', 
            'top_comment_1', and 'top_comment_1_classification'

        Returns
        -------
        list[dict]
            List of dictionaries containing query (concatenated submission text and title),
            top comment, and its classification for each entry
        """
        test_set = []
        for index, row in df.iterrows():
            test_set.append({
                'query': row['submission_text'] + '\n\n' + row['submission_title'], 
                'top_comment': row['top_comment_1'],
                'top_comment_classification': row['top_comment_1_classification']
            })

        return test_set    

    def create_balanced_test_set(self, df: pd.DataFrame, samples_per_class: int = 10) -> list[dict]:
        """
        Create a balanced test set with equal representation from each classification.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns 'submission_text', 'submission_title', 
            'top_comment_1', and 'top_comment_1_classification'
        samples_per_class : int, default=10
            Number of samples to include for each unique classification

        Returns
        -------
        list[dict]
            List of dictionaries containing balanced samples with query, top comment,
            and classification for each entry
        """
        balanced_df = (df.groupby('top_comment_1_classification')
                        .apply(lambda x: x.sample(n=min(len(x), samples_per_class)))
                        .reset_index(drop=True))
        
        test_set = []
        for _, row in balanced_df.iterrows():
           test_set.append({
                'query': row['submission_text'] + '\n\n' + row['submission_title'], 
                'top_comment': row['top_comment_1'],
                'top_comment_classification': row['top_comment_1_classification']
            })
        
        return test_set

    async def collect_responses(self, workflow: Workflow, test_set: list[tuple]):
        """
        Collect responses from the workflow for each sample in the test set.

        Parameters
        ----------
        workflow : Workflow
            The workflow object that generates responses
        test_set : list[tuple]
            List of test samples to process

        Returns
        -------
        list[dict]
            List of dictionaries containing the response, query, retrieved documents,
            top comment, and classification for each processed sample

        Notes
        -----
        This method handles exceptions for individual samples and continues processing
        the remaining samples if an error occurs
        """
        test_responses = []

        for i in tqdm(range(len(test_set)), desc='Generating agent responses on test set.'):
            try:
                result = await workflow.run(query=test_set[i]['query'])

                retrieved_doc_contents = []
                for node in result.source_nodes:
                    retrieved_doc_contents.append(node.text)
                response = ""
                async for chunk in result.async_response_gen():
                    response += chunk

                test_responses.append({
                    'response': response,
                    'query': test_set[i]['query'],
                    'retrieved_docs': retrieved_doc_contents,
                    'top_comment': test_set[i]['top_comment'],
                    'top_comment_classification': test_set[i]['top_comment_classification']
                })
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        return test_responses
    
    def parse_AITA_classification(
        self,
        response: str,
        parse_type: str = 'response'
    ) -> str:
            """
            Parse the AITA classification from a response string.

            Parameters
            ----------
            response : str
                The full response text to parse
            parse_type : str, default='response'
                The type of text to parse: 'response' or 'doc'

            Returns
            -------
            str
                The extracted classification or empty string if none found
            """
            AITA_classifications = ['NTA', 'YTA', 'NAH', 'ESH']

            if parse_type == 'response':
                pattern = '|'.join(AITA_classifications)
                match = re.search(pattern, response)

                if match:
                    return match.group(0)
                return ''
            
            elif parse_type == 'doc':
                # Search for AITA classifications only in the text after "Correct Justification:"
                justification_pos = response.find('Correct Classification:')

                if justification_pos != -1:
                    substring_after = response[justification_pos:]
                    pattern = '|'.join(AITA_classifications)
                    match = re.search(pattern, substring_after)
                    
                    if match:
                        return match.group(0)
                return ''
            
            else:
                raise ValueError(f"Invalid parse type: {parse_type}. Must be 'response' or 'doc'.")

    def evaluate(
        self,
        responses: list[dict],
        results_directory: str,
        classification_report_filepath: str,
        confusion_matrix_filepath: str,
        mcc_filepath: str,
        rouge_filepath: str,
        bleu_filepath: str,
        comet_filepath: str,
        retrieval_eval_filepath: str,
        retrieval_eval_summary_filepath: str
    ):
        """
        Evaluate both classifications and justifications from the responses.

        Parameters
        ----------
        responses : list[dict]
            List of response dictionaries containing predicted and true values
        results_directory : str
            Directory path where evaluation results will be saved
        classification_report_filepath : str
            Path to save classification report
        confusion_matrix_filepath : str
            Path to save confusion matrix visualization
        mcc_filepath : str
            Path to save Matthews Correlation Coefficient results
        rouge_filepath : str
            Path to save ROUGE scores
        bleu_filepath : str
            Path to save BLEU scores
        comet_filepath : str
            Path to save COMET scores

        Notes
        -----
        This method coordinates the evaluation of both classifications and justifications,
        saving results to the specified files
        """
        print('\nBEGIN EVALUATION\n')
        print('\tEvaluating classifications...')
        self.evaluate_classifications(
            responses=responses,
            results_directory=results_directory,
            classification_report_filepath=classification_report_filepath,
            confusion_matrix_filepath=confusion_matrix_filepath,
            mcc_filepath=mcc_filepath
        )

        print('\tEvaluating justifications...')
        self.evaluate_justifications(
            responses=responses,
            results_directory=results_directory,
            rouge_filepath=rouge_filepath,
            bleu_filepath=bleu_filepath,
            comet_filepath=comet_filepath
        )
        print('\tEvaluating Retrieval...')
        self.evaluate_retrieval(
            responses=responses,
            results_directory=results_directory,
            retrieval_eval_filepath=retrieval_eval_filepath,
            retrieval_eval_summary_filepath=retrieval_eval_summary_filepath
        )
        print('\nEND EVALUATION\n')
    
    def evaluate_classifications(
        self,
        responses: list[dict],
        results_directory: str,
        classification_report_filepath: str,
        confusion_matrix_filepath: str,
        mcc_filepath: str
    ):
        """
        Evaluate AITA classifications using multiple metrics.

        Parameters
        ----------
        responses : list[dict]
            List of response dictionaries containing predicted and true classifications
        results_directory : str
            Directory path where evaluation results will be saved
        classification_report_filepath : str
            Path to save classification report
        confusion_matrix_filepath : str
            Path to save confusion matrix visualization
        mcc_filepath : str
            Path to save Matthews Correlation Coefficient results

        Notes
        -----
        Generates and saves:
        - Classification report with precision, recall, and F1 scores
        - Confusion matrix visualization
        - Matthews Correlation Coefficient (MCC) score
        """

        for response in responses:
            response['predicted_classification'] = self.parse_AITA_classification(response['response'])

        true_labels = [response['top_comment_classification'] for response in responses]
        predicted_labels = [response['predicted_classification'] for response in responses]

        classification_metrics = classification_report(true_labels, predicted_labels, labels=self.AITA_classifications, zero_division=0)
        with open(os.path.join(results_directory, classification_report_filepath), 'w') as f:
            f.write(classification_metrics)

        cm = confusion_matrix(true_labels, predicted_labels, labels=self.AITA_classifications)
        plt.title(f'AITA Agent Classifications')
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.AITA_classifications, yticklabels=self.AITA_classifications, annot_kws={"size": 28})
        plt.xlabel('Predicted', fontsize=28)
        plt.ylabel('True', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.savefig(os.path.join(results_directory, confusion_matrix_filepath))

        matthews_metric = evaluate.load("matthews_correlation")
        mcc = matthews_metric.compute(references=[self.AITA_classifications.index(x) for x in true_labels], predictions=[self.AITA_classifications.index(x) for x in predicted_labels])
        with open(os.path.join(results_directory, mcc_filepath), 'w') as f:
            json.dump({'mcc': mcc}, f)
        
    def evaluate_justifications(
        self,
        responses: list[dict],
        results_directory: str,
        rouge_filepath: str,
        bleu_filepath: str,
        comet_filepath: str
    ):
        """
        Evaluate response justifications using multiple metrics.

        Parameters
        ----------
        responses : list[dict]
            List of response dictionaries containing predictions and references
        results_directory : str
            Directory path where evaluation results will be saved
        rouge_filepath : str
            Path to save ROUGE scores
        bleu_filepath : str
            Path to save BLEU scores
        comet_filepath : str
            Path to save COMET scores

        Notes
        -----
        Computes and saves:
        - ROUGE scores for measuring text similarity
        - BLEU scores for measuring translation quality
        - COMET scores for measuring text quality
        """
        predictions = [response['response'] for response in responses]
        references = [response['top_comment'] for response in responses]
        sources = [response['query'] for response in responses]

        # ROGUE scores
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)
        with open(os.path.join(results_directory, rouge_filepath), 'w') as f:
            json.dump(rouge, f)

        # BLEU scores
        bleu_metric = evaluate.load("bleu")
        bleu = bleu_metric.compute(predictions=predictions, references=references)
        with open(os.path.join(results_directory, bleu_filepath), 'w') as f:
            json.dump(bleu, f)

        # COMET scores
        comet_metric = evaluate.load('comet')
        comet_score = comet_metric.compute(predictions=predictions, references=references, sources=sources)
        with open(os.path.join(results_directory, comet_filepath), 'w') as f:
            json.dump(comet_score, f)

        # Toxicity analysis
        if torch.cuda.is_available():
            toxicity_model = pipeline("text-classification", 
                                      model="tomh/toxigen_roberta", 
                                      truncation=True, 
                                      device_map='cuda')
        else:
            toxicity_model = pipeline("text-classification", 
                                      model="tomh/toxigen_roberta", 
                                      truncation=True, 
                                      device_map='cpu')
        
        toxicity_scores = []
        for response in responses:
            toxicity_scores.append(toxicity_model(response['response'])[0])

        #######################
        # FINISH IMPLEMENTING #
        #######################
        

    def evaluate_retrieval(
        self,
        responses: list[dict],
        results_directory: str,
        retrieval_eval_filepath: str,
        retrieval_eval_summary_filepath: str
    ):
        """
        Evaluate the quality of document retrieval for AITA classifications.

        This method analyzes how well the retrieved documents align with the true classification
        of each query. It evaluates retrieval performance using multiple metrics including:
        - Top document classification accuracy
        - Classification distribution in retrieved documents
        - Ratio of retrieved documents matching the true classification

        Parameters
        ----------
        responses : list[dict]
            List of response dictionaries containing:
            - 'top_comment_classification': The true classification
            - 'retrieved_docs': List of retrieved document texts
        results_directory : str
            Directory path where evaluation results will be saved
        retrieval_eval_filepath : str
            Path to save detailed retrieval evaluation results for each query
        retrieval_eval_summary_filepath : str
            Path to save aggregated retrieval evaluation metrics

        Notes
        -----
        The method saves two JSON files:
        1. Detailed evaluation file containing per-query metrics:
            - true_classification: The correct AITA classification
            - top_doc_classification: Classification from the highest-ranked document
            - classification_counts: Distribution of classifications in retrieved docs
            - correct_classification_ratio: Proportion of docs with correct classification

        2. Summary file containing aggregate metrics:
            - top_doc_classification_accuracy: Accuracy of highest-ranked document
            - correct_classification_ratio: Average proportion of correctly classified docs
        """

        retrieval_evaluations = []

        for response in responses:
            # get the true classification for each response
            true_classification = response['top_comment_classification']

            # get the classifications for each retrieved document
            doc_classifications = []
            for doc in response['retrieved_docs']:
                doc_classifications.append(self.parse_AITA_classification(doc, parse_type='doc'))
            
            # get top retrieved doc classification
            top_doc_classification = doc_classifications[0]

            # get counts of each classification
            # initialize with all possible classifications using self.AITA_classifications
            classification_counts = {c: 0 for c in self.AITA_classifications}
            for classification in doc_classifications:
                classification_counts[classification] += 1
            
            # get ratio of classifications that match the true classification
            correct_classification_ratio = classification_counts[true_classification] / len(doc_classifications)

            # store results
            retrieval_evaluations.append({
                'true_class': true_classification,
                'top_doc_class': top_doc_classification,
                'class_counts': classification_counts,
                'doc_class_match_accuracy': correct_classification_ratio
            })

        # create summary results
        summary_results = {
            'avg_top_doc_class_match_accuracy': sum([x['top_doc_class'] == x['true_class'] for x in retrieval_evaluations]) / len(retrieval_evaluations),
            'avg_doc_class_match_accuracy': sum([x['doc_class_match_accuracy'] for x in retrieval_evaluations]) / len(retrieval_evaluations)
        }

        # save results
        with open(os.path.join(results_directory, retrieval_eval_filepath), 'w') as f:
            json.dump(retrieval_evaluations, f)
        
        with open(os.path.join(results_directory, retrieval_eval_summary_filepath), 'w') as f:
            json.dump(summary_results, f)