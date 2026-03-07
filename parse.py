from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


template = (
    "You are tasked with extracting specific information "
    "from the following text content: {dom_content}. "
    "Please follow these instructions carefully:\n\n"
    "1. Extract only information matching: {parse_description}\n"
    "2. Do not include explanations\n"
    "3. If nothing matches return empty string"
)


model = Ollama(model="gemma:2b")


def parse_with_ollama(dom_chunks, parse_description):

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):

        response = chain.invoke(
            {
                "dom_content": chunk,
                "parse_description": parse_description
            }
        )

        print(f"Parsed batch {i}")

        parsed_results.append(response)

    return "\n".join(parsed_results)