import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_core.documents import Document


import streamlit as st
import os

load_dotenv() 
pinecone_api_key = os.getenv("PINECONE_API_KEY")


def set_openai_api_key():
    if not st.session_state.get("openai_api_key"):
        st.error("Ajoute ta clé API OpenAI dans les paramètres pour continuer.")
        st.stop()
  
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        

def start_crawling(url, prefix, depth):
    visited_urls = set()
    wanted_urls = set()
    headers = {"User-Agent": "Mozilla/5.0"}
    
    
    def crawl_urls(url, prefix, depth):
        if depth < 0:
            return
        
        visited_urls.add(url)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            print(f"Exploration : {url}")

            # Extraite toutes les balises <a> (liens) de la page
            for anchor in soup.find_all('a'):
                href = anchor.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    print(f"Lien trouvé : {absolute_url}")

                    if absolute_url.startswith(prefix):
                        # Vérifier que l'URL ne contient pas de fragment (ex. : https://abc.com/blog/article#section)
                        parsed_url = urlparse(absolute_url)
                        if not parsed_url.fragment:
                            wanted_urls.add(absolute_url)
                    # Important : on ne suit que les URLs dans le préfixe (évite de sortir du domaine)        
                    if absolute_url not in visited_urls:
                        crawl_urls(absolute_url, prefix, depth -1)
                        
        except requests.exceptions.RequestException as e:
            print(f"Une erreur est survenue : {e}")
        
    crawl_urls(url, prefix, depth)
    return list(sorted(wanted_urls))
    
    
def generate_database(urls):
    # 1 - Récupérer le contenu textuel des pages web
    loaders = WebBaseLoader(
        web_paths=urls,
        requests_per_second=2,
        continue_on_failure=True,
        verify_ssl=True,
    )
    data = loaders.load()
    
    # 2 - Découper le texte en segments plus petits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
       
    # 3 - Convertir en embeddings et créer une base de données vectorielle
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    # 4 - Sauvegarder la base de données vectorielle en local
    db.save_local("faiss_index_pinecone")

    return db

def load_urls_to_documents(urls):
    documents = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # suppression scripts / styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose() 
                
            text = soup.get_text(separator=" ", strip=True)
            
            # nettoyage simple
            text = " ".join(text.split())
            
            if len(text) < 200:
                continue      # ignore pages trop pauvres
            
            documents.append( 
                             Document( 
                                      page_content=text,
                                      metadata={"source": url}
                                      )
                             )
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur URL {url} : {e}")
    
    return documents

def generate_pinecone_database(urls):
    # 1 - Récupérer le contenu textuel des pages web
    data = load_urls_to_documents(urls) 
    
    
    # 2 - Découper le texte en segments plus petits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
       
    # 3 - Convertir en embeddings et créer une base de données vectorielle
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_documents( 
                                                     documents=docs,
                                                     embedding=embeddings,
                                                     index_name="websitegpt"
                                                     )

    return vectorstore
    

  
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! Pose-moi ta question sur la base de connaissances."
                
            }
        ]
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


@st.cache_resource
def load_database(index_path="faiss_index_pinecone"):
    embeddings = OpenAIEmbeddings() 
    db = FAISS.load_local( 
                          index_path,
                          embeddings,
                          allow_dangerous_deserialization=True
                          )
    
    st.success("Base de données chargées avec succès !")
    
    return db


@st.cache_resource
def load_pinecone_database():
    embeddings = OpenAIEmbeddings() 
    
    return PineconeVectorStore.from_existing_index( 
                                                   index_name="websitegpt",
                                                   embedding=embeddings
                                                   )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def create_chain(_db):
    llm = ChatOpenAI( 
                     model="gpt-4.1-nano",
                     temperature=0.7,
                     streaming=True
                     )
    
    retriever = _db.as_retriever( 
                                 search_type="similarity",
                                 search_kwargs={"k": 4}
                                )
    
    prompt = ChatPromptTemplate.from_messages( 
                                              [
                                                  ( 
                                                        "system",
                                                        """Tu es un assistant expert et bienveillant.
                                                        Utilise UNIQUEMENT le contexte suivant pour répondre à la question. 
                                                        Si tu ne trouves pas la réponse dans le contexte, dis-le clairement.
                                                        Sois concis mais complet dans ta réponse. 
                                                        
                                                        Contexte: {context}"""),
                                                        MessagesPlaceholder("chat_history"),
                                                  
                                                        ("human", "{question}"
                                                    )
                                              ])
    
    chain = ( 
             {
                 "context": itemgetter("question") | retriever | format_docs,
                 "question": itemgetter("question"),
                 "chat_history": itemgetter("chat_history")
             }
             | prompt 
             | llm
             | StrOutputParser() 
             
             )
    
    return chain

    
def reset_conversation():
    st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour ! Pose-moi ta question sur la base de connaissances."
                
            }
        ]
        
    st.session_state.chat_history = []
    
                                                   
def get_conversation_stats():
    messages = st.session_state.get("messages", [])
    nb_questions = len([m for m in messages if m["role"] == "user"])
    nb_responses = len([m for m in messages if m["role"] == "assistant"]) - 1
    
    return { 
            "questions": nb_questions,
            "responses": nb_responses,
            "total_messages": len(messages) - 1
            }
                                             
    
    
