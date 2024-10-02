EthioMart Named Entity Recognition (NER) Fine-tuning Project
This project is part of the 10 Academy AI Mastery Program Week 5 Challenge. The goal is to fine-tune a Named Entity Recognition (NER) model for Amharic text, specifically focusing on extracting entities like product names, prices, and locations from e-commerce Telegram channels. The final solution helps EthioMart centralize decentralized e-commerce activities in Ethiopia by consolidating product data from multiple Telegram channels into one unified platform.

Project Structure
data/: Folder containing the preprocessed Amharic text data and the labeled dataset in CoNLL format.
fine_tuned_model/: Directory for saving the fine-tuned NER model and tokenizer.
ner_finetuning.ipynb: Jupyter notebook containing the entire code for the project.
README.md: This file, providing an overview of the project and steps.
Tasks Overview
Task 1: Data Ingestion and Preprocessing
We set up a data ingestion system to scrape messages from multiple Ethiopian-based Telegram channels. The scraped data includes:

Text messages in Amharic.
Metadata like timestamps, channel information, and message IDs.
Steps:

Scrape data from Telegram channels.
Preprocess the data by tokenizing, normalizing, and handling Amharic linguistic features.
Store the cleaned and structured data for NER model training.
Task 2: Data Labeling in CoNLL Format
To train the NER model, a subset of the data was labeled in CoNLL format. This format is suitable for Named Entity Recognition (NER) tasks.

Entities: Product names, locations, and prices.
Format: Each word is labeled as B-PRODUCT, I-PRODUCT, B-LOC, B-PRICE, etc.
We labeled 30-50 messages to create a training dataset.

Task 3: Fine-tuning NER Model
For NER, we fine-tuned the mBERT model using the labeled dataset. Fine-tuning steps include:

Loading the pre-labeled dataset in CoNLL format.
Tokenizing the data and aligning tokens with their corresponding labels.
Setting training parameters like learning rate, batch size, and epochs.
Using Hugging Face’s Trainer API to fine-tune the model.
Saving the trained model for deployment.
Results:

F1 Score: 0.942 on validation set.
Accuracy: 99.28% after 3 epochs.
Task 4: Model Comparison & Selection
We compared the fine-tuned mBERT model with other potential models like XLM-Roberta and DistilBERT to identify the best-performing model for the entity extraction task.

Steps:

Fine-tune multiple models.
Evaluate performance using precision, recall, F1 score, and accuracy.
Select the best-performing model (mBERT in this case).
Task 5: Model Interpretability
We implemented model interpretability tools to ensure transparency and understand how the NER model identifies entities.

Used SHAP (SHapley Additive Explanations) and LIME to explain model predictions.
Analyzed challenging cases (e.g., ambiguous text or overlapping entities).
Generated interpretability reports.
How to Run the Project
Prerequisites
Python 3.8+
Jupyter Notebook
Hugging Face Transformers
PyTorch (with GPU support for faster training)
Seqeval (for NER metrics evaluation)
Installation
Clone the repository:

git clone https://github.com/wubeabera123/Telegram-EthioMart-Entity-Extraction.git
cd Telegram-EthioMart-Entity-Extraction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Fine-tuning Process
Open the ner_finetuning.ipynb notebook.

Run each cell to:

Load and preprocess the dataset.
Fine-tune the NER model on the labeled Amharic data.
Save the fine-tuned model and tokenizer for deployment.
Model outputs and performance metrics will be displayed in the notebook.

Saving the Fine-tuned Model
Use the following function to save the model and tokenizer:

python
output_dir = "./fine_tuned_model"
save_model_and_tokenizer(model, tokenizer, output_dir)
The model will be saved in the fine_tuned_model/ directory for deployment.

Deliverables
Fine-tuned model: The fine-tuned mBERT model saved in the fine_tuned_model/ directory.
Data preprocessing steps: Documented in the notebook under Task 1.
NER model comparison and interpretability: Evaluation results for different models, including SHAP and LIME explanations.
Future Improvements
Increase the labeled dataset size to improve model accuracy further.
Experiment with more advanced multilingual models like AfroXLM-Roberta.
Enhance interpretability by incorporating more advanced explainability techniques.
Conclusion
This project successfully fine-tuned an NER model for Amharic text, capable of extracting product names, prices, and locations from Telegram e-commerce channels. By using SHAP and LIME, we ensured that the model’s predictions are interpretable, making it ready for production in a real-time data pipeline for EthioMart.