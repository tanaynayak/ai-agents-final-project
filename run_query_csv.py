import argparse
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embeddings import get_embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV file.")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv

    process_csv(input_csv, output_csv)


def process_csv(input_csv: str, output_csv: str):
    # Load the CSV file
    if not os.path.exists(input_csv):
        print(f"❌ Input file '{input_csv}' does not exist.")
        return

    data = pd.read_csv(input_csv)
    if "Q" not in data.columns:
        print("❌ Input CSV must contain a column named 'Q'.")
        return

    # Initialize embeddings and model
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = Ollama(model="gemma2")

    answers = []
    for index, row in data.iterrows():
        query_text = row["Q"]
        print(f"Processing query: {query_text}")

        try:
            response_text = query_rag(query_text, db, model)
            answers.append(response_text)
        except Exception as e:
            print(f"❌ Error processing query '{query_text}': {e}")
            answers.append("ERROR")

    # Add the answers to a new column
    data["modelname_answer"] = answers

    # Save the updated DataFrame to a new CSV
    data.to_csv(output_csv, index=False)
    print(f"✅ Results saved to '{output_csv}'")


def query_rag(query_text: str, db, model):
    # Search for similar documents in the database
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create a prompt from the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Get the model response
    response_text = model.invoke(prompt)

    return response_text


if __name__ == "__main__":
    main()