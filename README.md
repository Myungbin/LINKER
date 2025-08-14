# [UnderReview] LINKER: Leveraging Modality Knowledge for Semantic Relation Generation in Multimodal Recommendation

We propose LINKER, a novel framework designed to better learn users’ potential preference

## Requirements
```
conda create -n linker python=3.10
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy
pip install scipy
```

## 1. Prepare the Dataset and requirements
The preprocessed dataset is not included in this repository and needs to be prepared separately.
### Dataset
Amazon Review: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
### Data pre-processing
Amazon Review (5-core): Use the 5-core setting where each user and item has at least 5 reviews.  
Visual: Extract embeddings using pre-trained VGG16.  
Text: Extract embeddings using Sentence Transformer.  

## 2. Train & Evaluation
Run `python main.py` to train LINKER
```
cd src/
python main.py
```