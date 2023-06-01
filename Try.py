from transformers import pipeline
import PyPDF2
import os


def pdf_to_string(file_path):
    with open("SOURCE_DIRECTORY/"+file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text += page.extract_text()
            text = text.replace("\n", " ")
            data=text
    return data

def txt_to_string(file_path):
    with open("SOURCE_DIRECTORY/"+file_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data



nlp = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')





batch_size = 96
n_epochs = 2
base_LM_model = "roberta-base"
max_seq_len = 386
learning_rate = 3e-5
warmup_proportion = 0.3
doc_stride=128
max_query_length=64

paths=os.listdir("SOURCE_DIRECTORY")

a=""
for path in paths:

    if path.endswith(".txt"):
        data=txt_to_string(path)
        a=a+" "+data

    elif path.endswith(".pdf"):
        data=pdf_to_string(path)
        a=a+data




while True:
    qus=input("ask me: ")
    question_set = {
                    'question':qus, 
                    'context':a
                }

    results = nlp(question_set)

    print("")

    print("bot"+results['answer'])
    print("""------------------------------------------------------------------------
    """)
