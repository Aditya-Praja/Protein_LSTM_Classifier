Protein LSTM Classifier

Overview

This project implements a protein sequence classifier using a Long Short-Term Memory (LSTM) neural network built with PyTorch. The model learns to classify protein sequences into one of several functional categories based solely on their amino acid sequences.

The project was developed as a learning exercise to gain hands-on experience with:

* PyTorch
* Sequence modeling with LSTMs
* Building custom datasets
* Model training and evaluation
* Checkpointing
* Inference on unseen sequences
* Machine learning project organization

⸻

Project Structure

Protein_LSTM_Classifier/
│
├── data/
│   └── proteins.csv
│
├── models/
│   └── best_protein_lstm.pth
│
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── test_metrics.json
│
├── src/
│   ├── dataset.py
│   ├── evaluation.py
│   ├── model.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md

⸻

Dataset

The training data is stored as a CSV file.

Example format:

sequence	label
MKTLLVLLAVAVATASA	membrane
GVGKTAA…	enzyme
MTDKE…	dna_binding

Each row contains:

* a protein amino acid sequence
* its corresponding class label

⸻

Model Architecture

The classifier consists of four major components:

Protein Sequence
        │
        ▼
Embedding Layer
        │
        ▼
LSTM
        │
        ▼
Final Hidden State
        │
        ▼
Linear Classifier
        │
        ▼
Logits
        │
        ▼
Softmax (during prediction)

Embedding Layer

Converts amino acid tokens into dense vector representations.

LSTM

Processes the amino acid sequence while learning long-range relationships.

Linear Layer

Maps the final hidden state to one output for each protein class.

⸻

Training

Train the model using:

python train.py

Optional arguments:

python train.py \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 0.0005

During training the script:

* loads the dataset
* builds the vocabulary
* encodes labels
* creates train/validation/test splits
* trains the model
* evaluates on the validation set every epoch
* saves the best checkpoint
* generates loss and accuracy plots

⸻

Evaluation

Evaluate the saved model:

python evaluate.py

Evaluation reports:

* Test loss
* Test accuracy
* Classification report
* Confusion matrix
* Saved metrics

Outputs are stored inside:

results/

⸻

Prediction

Predict the class of a new protein sequence:

python predict.py \
    --sequence "MKTLLVLLAVAVATASA"

Example output:

Prediction results
Predicted class:
membrane
Confidence:
0.93
Class probabilities
membrane       0.93
enzyme         0.04
structural     0.02
dna_binding    0.01

⸻

Evaluation Metrics

The project reports:

* Accuracy
* Precision
* Recall
* F1-score
* Support
* Confusion Matrix

These metrics provide a more complete understanding of model performance than accuracy alone.

⸻

Checkpoint Contents

The saved checkpoint contains:

* model weights
* optimizer state
* epoch
* validation loss
* vocabulary
* label mapping
* embedding dimension
* hidden dimension

This allows the model to be reconstructed for evaluation and prediction.

⸻

Dependencies

Install the required packages:

pip install torch torchvision scikit-learn matplotlib pandas

or

pip install -r requirements.txt

⸻

Machine Learning Pipeline

Load Dataset
      │
      ▼
Create Vocabulary
      │
      ▼
Encode Labels
      │
      ▼
Split Dataset
      │
      ▼
Create DataLoaders
      │
      ▼
Train Model
      │
      ▼
Validate Each Epoch
      │
      ▼
Save Best Checkpoint
      │
      ▼
Evaluate on Test Set
      │
      ▼
Predict on New Sequences

⸻

Current Limitations

This project is intended as a learning implementation.

Current limitations include:

* Small demonstration dataset
* Single-layer LSTM architecture
* No early stopping
* No hyperparameter search
* Limited biological diversity in the training data

Because of these limitations, the model should not be used for biological research or clinical decision-making.

⸻

Future Improvements

Potential future enhancements include:

* Bidirectional LSTM
* Multi-layer LSTM
* Dropout regularization
* Larger curated protein datasets
* Hyperparameter tuning
* Class-weighted loss functions
* Transformer-based protein embeddings (e.g., ESM)
* Cross-validation
* Automated experiment tracking