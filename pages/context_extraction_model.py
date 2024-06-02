from transformers import BertTokenizerFast, BertForTokenClassification

import streamlit as st
import pandas as pd

import torch
import time


st.set_page_config("NLP App - Context extraction model", layout="wide")

with st.sidebar:
    st.title("NLP Project")
    st.page_link("app.py", label="App", icon="❌")
    st.page_link("./pages/text_detection_model.py", label="Text detection model", icon="❌")
    st.page_link("./pages/ocr_model.py", label="OCR model", icon="✔️")
    st.page_link("./pages/context_extraction_model.py", label="Context extraction", icon="✔️")

if "default" not in st.session_state:
    st.session_state["default"] = ""

if "testVisibility" not in st.session_state:
    st.session_state["testVisibility"] = False

st.title("Context Extraction Model (Bert)")


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

# --------------------------------------- Using the model ---------------------------------------------

BERT_PATH = "./ressources/Models/context_extraction"

# Mapping of entity labels to descriptive words
entity_labels = {
    "PERSON": "person",
    "NORP": "nationalities or religious or political groups",
    "FAC": "facility",
    "ORG": "organization",
    "GPE": "geopolitical entity",
    "LOC": "location",
    "PRODUCT": "product",
    "DATE": "date",
    "TIME": "time",
    "PERCENT": "percent",
    "MONEY": "money",
    "QUANTITY": "quantity",
    "ORDINAL": "ordinal",
    "CARDINAL": "cardinal",
    "EVENT": "event",
    "WORK_OF_ART": "work of art",
    "LAW": "law",
    "LANGUAGE": "language"
}

# Function to get the context explanation of a text
def get_context_explanation(text, model, tokenizer):
    # Tokenize the input text
    tokens = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Get the model's predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert token IDs to tokens
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    labels = predictions.squeeze().tolist()

    # Load the label names
    label_names = model.config.id2label
    
    # Extract named entities and their context
    entities = []
    current_entity = {"type": None, "tokens": []}

    for token, label in zip(tokens_list, labels):
        if token in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        
        label_name = label_names[label]
        
        if label_name.startswith("B-"):  # Beginning of a new entity
            if current_entity["tokens"]:
                entities.append(current_entity)
            current_entity = {"type": label_name[2:], "tokens": [token]}
        
        elif label_name.startswith("I-") and current_entity["type"] == label_name[2:]:
            current_entity["tokens"].append(token)
        
        else:
            if current_entity["tokens"]:
                entities.append(current_entity)
            current_entity = {"type": None, "tokens": []}
    
    if current_entity["tokens"]:
        entities.append(current_entity)

    # Generate the context explanation
    explanation = ["The text mentions several key entities."]
    for entity in entities:
        # Join tokens, handling subwords properly
        entity_tokens = [token.replace('##', '') if token.startswith('##') else token for token in entity["tokens"]]
        entity_text = tokenizer.convert_tokens_to_string(entity_tokens)
        entity_text = entity_text.replace(' - ', '-').strip()
        entity_type = entity_labels.get(entity['type'], entity['type'])
        
        explanation.append(f"{entity_text} is a {entity_type}.")
    
    return " ".join(explanation)

# Load the BERT-based NER model and tokenizer
def load_Bert():
    ner_model = BertForTokenClassification.from_pretrained(BERT_PATH)
    ner_tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
    return ner_model, ner_tokenizer


# -------------------------------------- Testing the model ---------------------------------------------

st.header("Testing the model")

importFrame = st.container()
outputFrame = st.container()

text_input = importFrame.text_area(
    "Enter some text:",
    value=st.session_state["default"]
)

testText = importFrame.button("Test Text", disabled=st.session_state["testVisibility"])

if testText:
    text = ''''''

    with open("./ressources/base_article.txt", "r", encoding='utf-8') as f:
        text = f.read()
    
    print(text)

    st.session_state["default"] = text
    st.session_state["testVisibility"] = True
    st.rerun()

if text_input:
    statusBar = importFrame.status("Processing...", expanded=True)

    ner_model, ner_tokenizer = load_Bert()

    context_explanation = get_context_explanation(text_input, ner_model, ner_tokenizer)

    statusBar.update(label="Process completed!", state="complete", expanded=False)
    
    outputFrame.subheader("Result")
    chat_output =  outputFrame.chat_message("ai")
    chat_output.write_stream(stream_data(context_explanation))


# ------------------------------------------- Dataset ---------------------------------------------------


datasetFrame = st.container()

datasetFrame.divider()

datasetFrame.header("Dataset")

datasetFrame.subheader("OntoNotes 5.0")

datasetFrame.markdown(
    '''<p><strong>OntoNotes 5.0</strong> is a large corpus comprising various genres of text (news, conversational telephone speech, weblogs, usenet newsgroups, broadcast, talk shows) in three languages (English, Chinese, and Arabic) with structural information (syntax and predicate argument structure) and shallow semantics (word sense linked to an ontology and coreference).</p>
<p>OntoNotes Release 5.0 contains the content of earlier releases - and adds source data from and/or additional annotations for, newswire, broadcast news, broadcast conversation, telephone conversation and web data in English and Chinese and newswire data in Arabic.</p>
<p>Here, we are using the english dataset, <i>english_v12</i> to be precise.</p>
<h5>Three first rows of the dataset</h5>
''',
    unsafe_allow_html=True
    )

dataset_extract = pd.read_csv('./ressources/ontonotes_v5_dataset_extract.csv', sep=";", header=0)
datasetFrame.dataframe(dataset_extract, hide_index=True, use_container_width=True)

datasetFrame.page_link("https://huggingface.co/datasets/conll2012_ontonotesv5/viewer/english_v12", label="Download the dataset")


# -------------------------------------- Fine tuning the model ---------------------------------------------

modelFrame = st.container()

modelFrame.divider()

modelFrame.header("Fine tuning the model")

modelFrame.subheader("Import the librairies")

modelFrame.markdown('''<p>We start by importing the necessary libraries, including transfomers for importing models and training them, and other utilities for data handling and visualization.</p>''',
                    unsafe_allow_html=True)

code = '''from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np

import evaluate
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define Constant")

code = '''label_all_tokens = True'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Load the dataset")

code = '''dataset = load_dataset('conll2012_ontonotesv5', 'english_v12', split='train')'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define label mappings")

code = '''labels_list = dataset.features['sentences'][0]['named_entities'].feature.names
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Initialize the tokenizer")

code = '''tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')'''

modelFrame.code(code, language='python', line_numbers=True)


modelFrame.subheader("Tokenize the dataset")

code = '''def tokenize_and_align_labels(examples):
    tokenized_inputs = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for sentences in examples['sentences']:
        for sentence in sentences:
            words = sentence['words']
            labels = sentence['named_entities']
            
            tokenized_input = tokenizer(words, truncation=True, is_split_into_words=True)
            
            aligned_labels = []
            word_ids = tokenized_input.word_ids()
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            
            tokenized_input["labels"] = aligned_labels
            
            tokenized_inputs['input_ids'].append(tokenized_input['input_ids'])
            tokenized_inputs['attention_mask'].append(tokenized_input['attention_mask'])
            tokenized_inputs['labels'].append(tokenized_input['labels'])

    return tokenized_inputs'''

modelFrame.code(code, language='python', line_numbers=True)

code = '''tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Load the pre-trained BERT model for token classification")

code = '''model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label2id))
model.config.id2label = id2label
model.config.label2id = label2id'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the data collator")

code = '''data_collator = DataCollatorForTokenClassification(tokenizer)'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the evaluation metric")

code = '''metric = evaluate.load("seqeval")'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the function for metric computation")

code = '''def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Define the training arguments")

code = '''training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Split the tokenized dataset into train and validation sets")

modelFrame.markdown('''<p>We use 90% of the dataset from training the model and the remaining 10% for validation.</p>''',
                    unsafe_allow_html=True)

code = '''train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Create the Trainer")

code = '''trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Train the model")

code = '''trainer.train()'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Evaluate the model")

code = '''trainer.evaluate()'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Save the model and tokenizer")

code = '''output_dir = "./saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Plot the training and evaluation loss")

code = '''history = trainer.state.log_history

train_loss = [entry['loss'] for entry in history if 'loss' in entry]
eval_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(1, (len(eval_loss)-1), num=len(train_loss)), train_loss, label='Training Loss')
plt.plot(np.linspace(1, (len(eval_loss)-1), num=len(eval_loss)), eval_loss, label='Evaluation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Over Epochs')
plt.legend()
plt.show()
'''

modelFrame.code(code, language='python', line_numbers=True)

modelFrame.subheader("Plot the evaluation Accuracy")

code = '''history = trainer.state.log_history

eval_accuracy = [entry['eval_accuracy'] for entry in history if 'eval_accuracy' in entry]

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(1, (len(eval_accuracy)-1), num=len(eval_accuracy)), eval_accuracy, label='Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Evaluation Accuracy Over Epochs')
plt.legend()
plt.show()
'''

modelFrame.code(code, language='python', line_numbers=True)

with open("./ressources/python_code/context_extraction_model.ipynb") as f:
    modelFrame.download_button("Download Notebook", f, "context_extraction_model.ipynb")

modelFrame.divider()

modelFrame.header("Training result")

col1, col2 = modelFrame.columns(2)

chart_data = pd.read_csv('./ressources/context_extraction_loss_results.csv', sep=";", header=0)
col1.line_chart(chart_data, x="Epoch", y=["train_loss"])

chart_data = pd.read_csv('./ressources/context_extraction_accuracy_results.csv', sep=";", header=0)
col2.line_chart(chart_data, x="Epoch", y=["accuracy"])

modelFrame.markdown('''<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Training time: </span><span style="color: #00af00; text-decoration-color: #00af00">2h</span> (with NVIDIA GTX 970)
</pre>''', unsafe_allow_html=True)






