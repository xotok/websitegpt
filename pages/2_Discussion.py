import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time
from utils import set_openai_api_key, init_session_state, load_database, create_chain, reset_conversation, get_conversation_stats, load_pinecone_database


st.set_page_config(
    page_title="Chat IA",
    page_icon="👉", 
    layout="wide"
    )

st.title("👉 Chat IA avec un site internet")
st.caption("Propulsé par LangChain et OpenAI")
st.markdown("---")

init_session_state()

with st.sidebar:
    st.header("Options")
    
    if st.button("Nouvelle conversation", use_container_width=True):
        reset_conversation() 
        st.rerun() 
    
    st.divider()
    
    st.subheader("Statistiques en temps réel")
    stats = get_conversation_stats() 
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric( 
                  label="Questions",
                  value=stats["questions"],
                  delta= None if stats["questions"] == 0 else "+1"
                  )
    with col2:
        st.metric( 
                  label="Réponses",
                  value=stats["responses"],
                  delta= None if stats["responses"] == 0 else "+1"
                  )
    
    st.divider() 
    
    st.subheader("A propos")
    st.caption("""
               Chat RAG Conversationnel
               
               Cette application combine :
               - Pinecone pour la recherche vectorielle
               - GPT-4.1-nano pour la génération
               - LanChain pour l'orchestration
               - Streamlit pour l'interface
               """)
    
    

set_openai_api_key() 
if st.session_state.openai_api_key:
    try:
        # Charger la base de données FAISS locale
        # local_db = load_database()
        
        # Charger la base de données Pinecone
        pinecone_db = load_pinecone_database() 
        st.success("Base de données chargées avec succès !")
   
        chain = create_chain(pinecone_db)
    except Exception as e:
        st.error("Impossible de charger la base de connaissances.")
        with st.expander("Voir l'erreur technique"):
            st.code(f"{type(e).__name__}: {str(e)}", language="python")
    
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if question := st.chat_input("Pose ta question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Hmm, bonne question. Un instant..."):
                try:
                    def response_generator(delay=0):
                        for chunk in chain.stream({ 
                                                   "question": question,
                                                   "chat_history": st.session_state.chat_history
                                                   }):
                            text = chunk if isinstance(chunk, str) else chunk.content
                            
                            for char in text:
                                yield char
                                time.sleep(delay)
                        
                    full_response = st.write_stream(response_generator(0.02))
                    
                    st.session_state.chat_history.append(HumanMessage(content=question))
                    st.session_state.chat_history.append(AIMessage(content=full_response))
                
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun() 

                except Exception as e:
                    st.error("L'IA est indisponible pour le moment...")
                    with st.expander("Voir l'erreur technique"):
                        st.code(f"{type(e).__name__}: {str(e)}", language="python")
                
    
    st.divider() 
    st.caption("Astuce : Plus ta question sera précise, meilleure sera la réponse !")
    
