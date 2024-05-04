# Translation Models with Metric Evaluation
> A comparison of multiple Machine Translation Models, including a fine-tuned model, using metric evaluation

### Setup

Install the required packages:

```
pip install "transformers[torch]"
pip install datasets

pip install torch --index-url https://download.pytorch.org/whl/cu118
or
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Information
This was a project completed for university as a group project which involved some task related to Natural Language Processing, in this case Machine Translation from French to English. In this, multiple models were compared with each other using metrics such as:
* BLEU (BiLingual Evaluation Understudy)
* ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
* METEOR (Metric for Evaluation of Translation with Explicit ORdering).

Additionally, a fine-tuned model was developed using the T5-Small model to perform translations from French to English. This fine-tuned model was trained using data obtained from this Kaggle dataset

https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset

Metric evaluation was done using the same subset of the Kaggle dataset.
