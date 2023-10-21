import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
import json
from streamlit_lottie import st_lottie
from dotenv import load_dotenv


st.set_page_config(layout="wide",page_title="Chat PDF App", page_icon="page_icon.jpg", initial_sidebar_state="expanded")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
}

with st.sidebar:
       st.title('Interact with PDF🦜')
       st.markdown('''
       ## About⚙️
       This App helps you to interact with your PDFs.
                   ''')
       #add_vertical_space(1)
       st.write('Made with 💚 by Bibhuti Baibhav Borah')

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def main():

       lottie_ani = load_lottiefile("animation.json")
       st_lottie(
        lottie_ani,
        speed =1,
        reverse=False,
        quality= "high",
        width=340)

       st.header("Chat With PDF🔎")

       load_dotenv()



       pdf = st.file_uploader("#### Please Upload the PDF👇🏻", type='pdf')
       if pdf is not None:
              pdf_reader1 = PdfReader(pdf)
             
              text = ""
              for page in pdf_reader1.pages:
                     text += page.extract_text()
              
              text_splitter = RecursiveCharacterTextSplitter(
                     chunk_size = 1000,
                     chunk_overlap = 200,
                     length_function=len
              )
              chunks = text_splitter.split_text(text=text)
              #st.write(chunks)

              store_name = pdf.name[:-4]

              if os.path.exists(f"{store_name}.pkl"):
                      with open(f"{store_name}.pkl","rb") as f:
                             VectorStore = pickle.load(f)

              else:
                     embeddings = OpenAIEmbeddings()
                     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                     with open(f"{store_name}.pkl","wb") as f:
                            pickle.dump(VectorStore,f)


              #st.write(text) 
              query = st.text_input("Ask questions about your PDF file")  
              #st.write(query)


              if query:
                     docs = VectorStore.similarity_search(query=query, k=3)
                     llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                     chain = load_qa_chain(llm=llm, chain_type="stuff")
                     with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=query)
                     st.write(response)

if __name__ == '__main__':
       main()      

