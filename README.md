# Sentiment analysis on 160M tweets with BERT and DistilBERT:
In this repo, we provide an experiment on Sent140 dataset with mislabeled data featuring:

- **Sent140** binary classification problem
- Compare **γ-logistic loss** with conventional **cross entropy**
- BERT-base and DistilBERT-base as model

## Usage
1. **Prepare data:**
     - Run `preprocess.ipynb` file and prepare the data stored as `/dataset/train.csv`, `/dataset/val.csv`, and `/dataset/test.csv`:
     - <img src=/images/wordcloud_fig.png width=100% height=100%>
     
2. **Run experiments (training):**
     - Train with BERT:
       - Without mislabel data:
          - Run `python main.py --model_name='BERT'`
       - With mislabel data:
          - Run `python main.py ---model_name='BERT' -mislabel_rate=float(0-1)`

     - Train with DistilBERT:
       - Without mislabel data:
          - Run `python main.py --model_name='DistilBERT'`
       - With mislabel data:
          - Run `python main.py --model_name='DistilBERT' --mislabel_rate=float(0-1)`
3. **Run experiments (testing):**
     - Run `python inference.py`.
4. **Output:**
     - ROC curve, confusion matrix, prediction report csv are saved in `outputs/logs/{exp_name}/`.
     <img src=/images/sent_results.png width=75% height=75%>
