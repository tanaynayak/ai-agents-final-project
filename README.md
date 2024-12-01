# CourseMate: Your Personalized Academic Virtual Assistant

CourseMate is a Retrieval-Augmented Generation (RAG)-driven virtual assistant designed to provide personalized academic support through efficient knowledge retrieval and contextual response generation.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Features

- **Personalized Support**: Tailors responses to individual academic queries.
- **Knowledge Retrieval**: Utilizes advanced models to fetch relevant information.
- **Contextual Responses**: Generates answers that are contextually appropriate to your questions.
- **Easy Integration**: Simple setup process to get you started quickly.

## Prerequisites

- **macOS**
- **[Ollama](https://ollama.com/download/mac)** installed
- **Python 3.x**
- **pip**

## Installation

1. **Install Ollama for macOS**

   Download and install Ollama from [here](https://ollama.com/download/mac).

2. **Download Required Models**

   Open your terminal and run:

   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   ```

3. **Install Python Dependencies**

    ```python
    pip install -r requirements.txt
    ```

## Usage

1. **Create The Database**

    Initialize the knowledge database by running:

    ```python
    python create_database.py
    ```

2. **Run Queries**

    ```python
    python run_query.py 'Your question here'
    ```


