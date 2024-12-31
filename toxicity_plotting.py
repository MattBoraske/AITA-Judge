from transformers import pipeline
import matplotlib.pyplot as plt
import torch
from multiprocessing import freeze_support
import os
import random
from pprint import pprint
import random

def get_toxicity_scores(responses: list[str]):
    # load toxigen roberta model
    if torch.cuda.is_available():
        toxigen_roberta = pipeline("text-classification", 
                                 model="tomh/toxigen_roberta", 
                                 truncation=True, 
                                 device_map='cuda')
    else:
        toxigen_roberta = pipeline("text-classification", 
                                 model="tomh/toxigen_roberta", 
                                 truncation=True, 
                                 device_map='cpu')

    scores = toxigen_roberta(responses)
    return scores


def plot_toxicity_scores(scores, bins=40, figsize=(12, 7)):
    """
    Create an improved histogram of toxicity scores with better visualization and labeling.
    
    Args:
        scores: List of dictionaries containing toxicity scores and labels
        bins: Number of bins for the histogram
        figsize: Tuple specifying figure dimensions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert scores to a format where toxic is negative and benign is positive
    processed_scores = []
    for s in scores:
        if s['label'] == 'TOXIC':
            processed_scores.append(-s['score'])  # Make toxic scores negative
        else:
            processed_scores.append(s['score'])   # Keep benign scores positive
    
    # Create the figure and axis
    plt.figure(figsize=figsize)
    
    # Create histogram
    n, bins, patches = plt.hist(
        processed_scores,
        bins=bins,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7,
        color='skyblue'
    )
    
    # Add a kernel density estimate
    density = plt.gca().twinx()
    kde_xs = np.linspace(-1, 1, 200)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(processed_scores)
    density.plot(kde_xs, kde.evaluate(kde_xs), 'r-', lw=2, alpha=0.5)
    density.set_ylim(0, max(kde.evaluate(kde_xs)) * 1.2)
    density.set_ylabel('Density', color='red')
    density.tick_params(axis='y', colors='red')
    
    # Customize the plot
    plt.gca().set_xlabel('Toxicity Score', fontsize=12, fontweight='bold')
    plt.gca().set_ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Distribution of Text Toxicity Scores', fontsize=14, pad=20)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # Set x-axis limits and ticks
    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1, 1.2, 0.2))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add text labels for toxic and benign regions
    plt.text(-0.9, plt.gca().get_ylim()[1]*0.95, 'TOXIC', fontsize=10, color='red')
    plt.text(0.1, plt.gca().get_ylim()[1]*0.95, 'BENIGN', fontsize=10, color='green')
    
    # Add statistics
    mean_score = np.mean(processed_scores)
    median_score = np.median(processed_scores)
    plt.axvline(x=mean_score, color='green', linestyle=':', label=f'Mean ({mean_score:.2f})')
    plt.axvline(x=median_score, color='purple', linestyle=':', label=f'Median ({median_score:.2f})')
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('toxicity_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
if __name__ == '__main__':
    freeze_support()

    sentences = [
        "The cat sleeps on the window sill.",
        "She drinks coffee every morning.",
        "The rain falls softly outside.",
        "They played cards all night.",
        "The book sits on the shelf.",
        "He runs in the park daily.",
        "The flowers bloom in spring.",
        "Birds sing in the morning.",
        "The child draws with crayons.",
        "We watched the sunset together.",
        "The dog chases its tail.",
        "The phone rings loudly.",
    ]

    # Your existing code to generate scores
    #scores = get_toxicity_scores(sentences)

    # Generate sample data
    scores = []
    # Add some LABEL_0 scores (mostly near 1)
    for _ in range(100):
        scores.append({
            'label': 'BENIGN',
            'score': random.gauss(0.9, 0.1)
        })
    # Add some more LABEL_0 scores (mostly near 0)
    for _ in range(100):
        scores.append({
            'label': 'BENIGN',
            'score': random.gauss(0.1, 0.1)
        })
    # Add some LABEL_1 scores (mostly near 1, will be converted to -1 in plot)
    for _ in range(100):
        scores.append({
            'label': 'TOXIC',
            'score': random.gauss(0.9, 0.1)
        })
    # Add some more LABEL_1 scores (mostly near 0, will be converted to -0 in plot)
    for _ in range(100):
        scores.append({
            'label': 'TOXIC',
            'score': random.gauss(0.1, 0.1)
        })
        
    # Plot with improved visualization
    plot_toxicity_scores(scores, bins=30)