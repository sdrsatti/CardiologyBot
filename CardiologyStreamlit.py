import streamlit as st 

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


OPENAI_API_KEY = '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'
PINECONE_API_KEY = '518effcc-8f4d-4c7c-981f-b1a63bd90362'
PINECONE_API_ENV = 'us-east4-gcp'




    
def getanswer(question):
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 
    index_name = "langchain2"
    
    #docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    query = question
    docs = docsearch.similarity_search(query, include_metadata=True)
    answer = chain.run(input_documents=docs, question=query)

    return(answer)

st.set_page_config(page_title='Cardiology ChatGPT', page_icon=':robot')

st.header('Cardiology ChatGPT')
st.write('')
input_prompt = st.text_area(label='What is your query:', key='user_input')

if st.button(label='Answer'):
    
    answer = getanswer(input_prompt)
    st.write(answer)
    



