import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Response
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
import pinecone
from utils.template import prompt_template, template
import evaluate
import uvicorn
import os

app = FastAPI()

api_key = os.environ.get("OPENAI_API_KEY")

# pinecone 

llm = OpenAI(temperature=0, model_name='text-davinci-003',max_tokens=500)

PINECONE_API_KEY = '9ffa659d-198e-4658-b839-efe1a9c801a6'

PINECONE_ENV = 'asia-northeast1-gcp'

squad_metric = evaluate.load("squad")

embeddings = OpenAIEmbeddings()

pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENV  # next to api key in console
)
index_name = "bank-live-promo"

db = Pinecone.from_existing_index(embedding=embeddings,index_name=index_name)
chain = LLMChain(llm=llm,prompt=prompt_template)

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
        docs = db.similarity_search(j['question'],k=2)
        data[i]['CONTEXT'] = docs
    
    

    # Perform evaluation
    # Replace the following code with your evaluation logic
    # ...
    prediction = chain.apply(data)
    eval_chain = QAEvalChain.from_llm(llm)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
