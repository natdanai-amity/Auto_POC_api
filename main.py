import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Response
from langchain.evaluation.qa import QAEvalChain
import numpy
from utils.template import prompt_template, template
import evaluate
import uvicorn
import os
from utils.template import prompt_template,template,system_message_prompt,human_message_prompt
from similarity_search import search_documents
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils.translate import thai2en,en2thai

load_dotenv()
app = FastAPI()
openai_api_key = os.getenv("OPENAI_API_KEY")

# api_key = os.environ.get("OPENAI_API_KEY")

# pinecone 

squad_metric = evaluate.load("squad")

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# chat_prompt = ChatPromptTemplate.from_messages([prompt_template])
chat = ChatOpenAI(temperature=0)
chain = LLMChain(llm=chat,prompt=chat_prompt)

squad_metric = evaluate.load("squad")

def get_query(question):
    question = thai2en(question)
    context = search_documents(question, 5)
    context_all = [i['content'] for i in context]
    return context_all, question

@app.get("/")
async def home():
    return {"Amity":"Validation"}

@app.post("/evaluate")
async def evaluate_excel_file(file: UploadFile = File(...)):
    # Read Excel file
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))

    data = []
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        data.append({'question': question, 'answer': answer})

    #similarity search
    for i,j in enumerate(data):
        try:
            docs = search_documents(thai2en(j['question']),k=2)
            data[i]['CONTEXT'] = docs
        except:
            print("too long")

    # Perform evaluation
    # Replace the following code with your evaluation logic
    # ...
    prediction = chain.apply(data)
    eval_chain = QAEvalChain.from_llm(chat)
    graded_outputs = eval_chain.evaluate(data, prediction, question_key="question", prediction_key="text")
    # Add more code or actions here based on the button click
    # Display graded outputs
    graded_texts = [output["text"] for output in graded_outputs]
    df['graded_output'] = graded_texts
    df['gpt-answer'] = [i['text'] for i in prediction]

    results = []
    for i in range(len(df)):
        references = [{'answers': {'answer_start': [0], 'text': [df['answer'][i]]}, 'id': '1'}]
        gpt_gen = [{'prediction_text': df['gpt-answer'][i], 'id': '1'}]
        results.append(squad_metric.compute(predictions=gpt_gen, references=references)['f1'])
    print("Hello 2")
    df['confident'] = results

    # Return the evaluated results as a DataFrame

    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    # Return the Excel file as a downloadable attachment
    excel_file.seek(0)
    content = excel_file.getvalue()
    return Response(content=content, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={'Content-Disposition': 'attachment; filename=Result.xlsx'})
    # return df

