# Auto_POC_api
This repository is validation system of KM platform which is pass excel through api and return it

# prompts_validator
This is an auto validator api for check two of gpt generated
# FastAPI Application Dockerization

This repository contains a FastAPI application that can be containerized using Docker.

## Prerequisites

- Docker installed on your machine.

## Getting Started

1. Clone this repository to your local machine.

```bash
git clone <repository-url>
```
2. Navigate to the project directory.
```bash
cd <project-directory>
```
3. Create a .env file in the project directory and add your OpenAI API key as an environment variable
```bash
OPENAI_API_KEY=your_openai_api_key
```
4. Build the Docker image.
```bash
docker build -t manual-validator-app .
```
5. Run the Docker container.
```bash
docker run -p 8000:8000 manual-validator-app
```
6. Access the FastAPI application in your browser.
Open your browser and navigate to http://localhost:8000. You should see the FastAPI application running.

## How to pass excel file to API

```bash
curl -X 'POST' \
  '<API url>/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<excel path>;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ```
  ### Pass excel with python
  ```bash
import requests

url = 'http://localhost:8000/evaluate'
files = {'file': open('<excel path>', 'rb')}
headers = {'Accept': 'application/json'}

response = requests.post(url, files=files, headers=headers)
```
