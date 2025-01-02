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
import numpy as np

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
        toxicity_stats_filepath: str,
        toxicity_plot_filepath: str,
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
        toxicity_stats_filepath: str
            Path to save toxicity scores
        toxicity_plot_filepath: str
            Path to save toxicity plot
        retrieval_eval_filepath : str
            Path to save detailed retrieval evaluation results for each query
        retrieval_eval_summary_filepath : str
            Path to save aggregated retrieval evaluation metrics

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
            comet_filepath=comet_filepath,
            toxicity_stats_filepath=toxicity_stats_filepath,
            toxicity_plot_filepath=toxicity_plot_filepath
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
        comet_filepath: str,
        toxicity_stats_filepath: str,
        toxicity_plot_filepath: str
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
        toxicity_stats_filepath: str
            Path to save toxicity scores
        toxicity_plot_filepath: str
            Path to save toxicity plot

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
        #comet_metric = evaluate.load('comet')
        #comet_score = comet_metric.compute(predictions=predictions, references=references, sources=sources, batch_size=8)
        #with open(os.path.join(results_directory, comet_filepath), 'w') as f:
        #    json.dump(comet_score, f)

        # Toxicity analysis
        if torch.cuda.is_available():
            print('Downloading toxicity classification model onto GPU...')
            toxicity_model = pipeline("text-classification", 
                                      model="tomh/toxigen_roberta", 
                                      truncation=True, 
                                      device_map='cuda')
        else:
            print('Downloading toxicity classification model onto CPU...')
            toxicity_model = pipeline("text-classification", 
                                      model="tomh/toxigen_roberta", 
                                      truncation=True, 
                                      device_map='cpu')
        
        for response in responses:
            predicted_response_score = toxicity_model(response['response'])[0]
            if predicted_response_score['label'] == 'LABEL_0':
                predicted_response_score['label'] = 'BENIGN'
            else:
                predicted_response_score['label'] = 'TOXIC'
            response['predicted_response_toxicity'] = predicted_response_score

            true_response_score = toxicity_model(response['top_comment'])[0]
            if true_response_score['label'] == 'LABEL_0':
                true_response_score['label'] = 'BENIGN'
            else:
                true_response_score['label'] = 'TOXIC'
            response['true_response_toxicity'] = true_response_score

        predicted_toxicity_scores = [{'score': response['predicted_response_toxicity']['score'], 'label': response['predicted_response_toxicity']['label']} for response in responses]
        true_toxicity_scores = [{'score': response['true_response_toxicity']['score'], 'label': response['true_response_toxicity']['label']} for response in responses]

        predicted_toxicity_scores = [
            -s['score'] if s['label'] == 'TOXIC' else s['score'] 
            for s in predicted_toxicity_scores
        ]
        true_toxicity_scores = [
            -s['score'] if s['label'] == 'TOXIC' else s['score']
            for s in true_toxicity_scores
        ]

        # compute toxicity stats
        toxicity_stats = self.get_toxicity_stats(predicted_toxicity_scores, true_toxicity_scores)

        # save toxicity stats
        with open(os.path.join(results_directory, toxicity_stats_filepath), 'w') as f:
            json.dump(toxicity_stats, f)

        self.plot_toxicity_scores(
            predicted_toxicity_scores,
            true_toxicity_scores,
            toxicity_plot_filepath=os.path.join(results_directory, toxicity_plot_filepath),
            toxicity_stats=toxicity_stats,
        )

    def get_toxicity_stats(
            self,
            predicted_toxicity_scores: list[dict],
            true_toxicity_scores: list[dict]
        ) -> dict:
            """
            Compute statistics for a list of toxicity scores.
            """

            predicted_mean_score = round(np.mean(predicted_toxicity_scores), 3)
            predicted_median_score = round(np.median(predicted_toxicity_scores), 3)
            predicted_benign_count = sum(1 for s in predicted_toxicity_scores if s >= 0)
            predicted_toxic_count = sum(1 for s in predicted_toxicity_scores if s < 0)

            true_mean_score = round(np.mean(true_toxicity_scores), 3)
            true_median_score = round(np.median(true_toxicity_scores), 3)
            true_benign_count = sum(1 for s in true_toxicity_scores if s >= 0)
            true_toxic_count = sum(1 for s in true_toxicity_scores if s < 0)

            percent_change_mean = round(((predicted_mean_score - true_mean_score) / true_mean_score) * 100, 2)
            percent_change_median = round(((predicted_median_score - true_median_score) / true_median_score) * 100, 2)
            total_change_benign = predicted_benign_count - true_benign_count
            total_change_toxic = predicted_toxic_count - true_toxic_count
            percent_change_benign = 'N/A' if true_benign_count == 0 else round(((predicted_benign_count - true_benign_count) / true_benign_count) * 100, 2)
            percent_change_toxic = 'N/A' if true_toxic_count == 0 else round(((predicted_toxic_count - true_toxic_count) / true_toxic_count) * 100, 2)

            return {
                'predicted_response_toxicity_stats': 
                    {
                        'mean_score': predicted_mean_score,
                        'median_score': predicted_median_score,
                        'benign_count': predicted_benign_count,
                        'toxic_count': predicted_toxic_count,
                    },
                'true_response_toxicity_stats':
                    {
                        'mean_score': true_mean_score,
                        'median_score': true_median_score,
                        'benign_count': true_benign_count,
                        'toxic_count': true_toxic_count,
                    },
                'change_stats:':
                    {
                        'percent_change_mean': percent_change_mean,
                        'percent_change_median': percent_change_median,
                        'total_change_benign': total_change_benign,
                        'total_change_toxic': total_change_toxic,
                        'percent_change_benign': percent_change_benign,
                        'percent_change_toxic': percent_change_toxic,
                    }
            }

    def plot_toxicity_scores(
        self,
        predicted_toxicity_scores: list[dict],
        true_toxicity_scores: list[dict],
        toxicity_plot_filepath: str,
        toxicity_stats = dict[dict],
        bins: int = 40,
        figsize: tuple = (20, 7),
        labels: str =("Predicted Responses", "True Responses"),
    ):
        """
        Create an improved histogram of toxicity scores with better visualization and labeling.
        
        Args:
            predicted_toxicity_scores: List of dictionaries containing toxicity scores and labels for predicted responses
            true_toxicity_scores: List of dictionaries containing toxicity scores and labels for true responses
            toxicity_plot_filepath: Path to save toxicity plot
            toxicity_stats: Dictionary containing statistics for predicted responses
            bins: Number of bins for the histogram
            figsize: Tuple specifying figure dimensions
            labels: Tuple of strings for labeling the two datasets
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
                
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors for each dataset
        colors = {
            'hist1': 'skyblue',
            'hist2': 'lightgreen',
        }
        
        # Create histograms
        plt.hist(
            predicted_toxicity_scores,
            bins=bins,
            edgecolor='black',
            linewidth=1.2,
            alpha=0.5,
            color=colors['hist1'],
            label=f'Histogram ({labels[0]})'
        )
        
        plt.hist(
            true_toxicity_scores,
            bins=bins,
            edgecolor='black',
            linewidth=1.2,
            alpha=0.5,
            color=colors['hist2'],
            label=f'Histogram ({labels[1]})'
        )
        
        # Customize the plot
        ax.set_xlabel('Confidence Score', fontsize=14, labelpad=10)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Confidence Scores for Response Toxicity', fontsize=16, pad=20)
        
        # Add vertical line at 0
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # Set x-axis limits and ticks
        plt.xlim(-1, 1)
        plt.xticks(np.arange(-1, 1.2, 0.2))
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add text labels for toxic and benign regions
        plt.text(-0.6, plt.gca().get_ylim()[1]*0.95, 'TOXIC', fontsize=14)
        plt.text(0.4, plt.gca().get_ylim()[1]*0.95, 'BENIGN', fontsize=14)
        
        # Create two separate legends
        # First legend for dataset 1
        legend1_elements = [
            Patch(facecolor=colors['hist1'], alpha=0.5, edgecolor='black', label=labels[0]),
            Patch(facecolor='none', label=f"Mean: {toxicity_stats['predicted_response_toxicity_stats']['mean_score']:.2f}"),
            Patch(facecolor='none', label=f"Median: {toxicity_stats['predicted_response_toxicity_stats']['median_score']:.2f}"),
            Patch(facecolor='none', label=f"Benign Responses: {toxicity_stats['predicted_response_toxicity_stats']['benign_count']}"),
            Patch(facecolor='none', label=f"Toxic Responses: {toxicity_stats['predicted_response_toxicity_stats']['toxic_count']}")
        ]
        
        # Second legend for dataset 2
        legend2_elements = [
            Patch(facecolor=colors['hist2'], alpha=0.5, edgecolor='black', label=labels[1]),
            Patch(facecolor='none', label=f"Mean: {toxicity_stats['true_response_toxicity_stats']['mean_score']:.2f}"),
            Patch(facecolor='none', label=f"Median: {toxicity_stats['true_response_toxicity_stats']['median_score']:.2f}"),
            Patch(facecolor='none', label=f"Benign Responses: {toxicity_stats['true_response_toxicity_stats']['benign_count']}"),
            Patch(facecolor='none', label=f"Toxic Responses: {toxicity_stats['true_response_toxicity_stats']['toxic_count']}")
        ]
        
        # Add the legends with reduced spacing and closer to the plot
        legend1 = plt.legend(handles=legend1_elements, bbox_to_anchor=(1.02, 1), loc='upper left', prop={'size': 12})  # Changed from 1.15 to 1.02
        plt.gca().add_artist(legend1)
        
        #legend2 = plt.legend(handles=legend2_elements, bbox_to_anchor=(1.02, 0.75), loc='upper left')  # Changed from 1.15 to 1.02
        #plt.gca().add_artist(legend2)
        plt.legend(handles=legend2_elements, bbox_to_anchor=(1.02, 0.75), loc='upper left', prop={'size': 12})  # Changed from 1.15 to 1.02
        
        # Add explanatory note below legends
        plt.figtext(0.7475, 0.49, '*Negative scores indicate\ntoxic classifications while\n positive ones are benign.', fontsize=12, ha='center', va='top')
                
        # Adjust layout with modified spacing
        plt.tight_layout(rect=[0, 0, 0.82, 1])  # Changed from 0.85 to 0.82
        
        # Save the plot
        plt.savefig(toxicity_plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()  

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