import streamlit as st
from utils import start_crawling,set_openai_api_key,generate_pinecone_database


st.set_page_config(page_title="Scrape ton site web", page_icon="🖥", layout="centered")

st.title("Scrape ton site web")
st.markdown("Transforme le contenu d'un site web en base de connaissances intelligentes.")
st.markdown("---")

user_api_key = st.sidebar.text_input( 
                                     label="Clé API OpenAI",
                                     placeholder="sk-xxxxx",
                                     type="password",
                                     value=st.session_state.get("openai_api_key", "") or ""
                                     )

if user_api_key:
    st.session_state.openai_api_key = user_api_key
    set_openai_api_key()
      
    st.subheader("Paramètres du site à analyser")
   
    website_url = st.text_input( 
                                label="URL du site",
                                placeholder="Colle l'URL du site ici..."
                                ) 
    prefix_url = st.text_input( 
                                label="URL préfixe (filtre)",
                                placeholder="Ex : https://monsite.com/blog"
                                )
    depth = st.slider("Niveau de profondeur d'exploration", 0, 5, 1)
    
    if website_url and prefix_url:
        if st.button("Lancer l'exploration"):
            # Etape 1 - Récupérer tous les URLs souhaités
            with st.spinner("Exploration des URLs en cours..."):
                scraped_urls = start_crawling(website_url, prefix_url, depth)
            
            st.success(f"{len(scraped_urls)} URLs trouvés avec succès !")
        
            with st.expander("Voir les URLs détectés"):
                for i, url in enumerate(sorted(scraped_urls), start=1):
                    st.write(f"{i}. {url}")
    
            # Etape 2 - Générer une base de données vectorielle à partir du contenu textuel des URLs ci-dessus
            st.markdown("---")
            st.subheader("Création de la base vectorielle")
            with st.spinner("Analyse des sources et préparation de l'IA..."):
                try:
                    # Pour FAISS
                    # generate_database(scraped_urls)
                    
                    # Pour Pinecone
                    generate_pinecone_database(scraped_urls)
                    st.success("Assistant IA prêt ! Tu peux maintenant poser tes questions.")
                except Exception as e:
                    st.error("Impossible de construire la base de connaissances.")
                    with st.expander("Voir l'erreur technique"):
                        st.code(f"{type(e).__name__}: {str(e)}", language="python")
                        
                    
                                               
else:
    st.info("Ajoute ta clé API OpenAI dans la barre latérale pour commencer.")
