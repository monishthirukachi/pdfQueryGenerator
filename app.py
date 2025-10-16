# # streamlit_app.py
# import streamlit as st
# from dotenv import load_dotenv
# import os
# from PyPDF2 import PdfReader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# st.set_page_config(page_title="PDF QA with Google Gemini", layout="wide")
# st.title("PDF Question-Answering with Google Gemini Embeddings")

# # Upload PDF
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file:
#     pdf = PdfReader(uploaded_file)
#     st.write(f"Number of pages: {len(pdf.pages)}")
    
#     # Extract text
#     raw_text = ''
#     for page in pdf.pages:
#         text = page.extract_text()
#         if text:
#             raw_text += text
    
#     st.write(f"Extracted {len(raw_text)} characters from PDF.")

#     # Split text into chunks
#     text_splitter = CharacterTextSplitter(
#         separator=".",
#         chunk_size=800,
#         chunk_overlap=200,
#         length_function=len
#     )
#     texts = text_splitter.split_text(raw_text)
#     st.write(f"Split into {len(texts)} text chunks.")

#     # Create embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
#     docsearch = FAISS.from_texts(texts, embeddings)
#     st.success("Embeddings created!")

#     # Input for question
#     query = st.text_input("Ask a question about the PDF:")

#     if query:
#         model = GoogleGenerativeAI(
#             model="gemini-2.5-flash",
#             google_api_key=api_key
#         )
#         chain = load_qa_chain(model, chain_type="stuff")
#         docs = docsearch.similarity_search(query)
#         answer = chain.run(input_documents=docs, question=query)
#         st.subheader("Answer:")
#         st.write(answer)



# streamlit_app.py
# import streamlit as st 
# from dotenv import load_dotenv 
# import os 
# from PyPDF2 import PdfReader 
# from langchain_text_splitters import CharacterTextSplitter 
# from langchain_community.vectorstores import FAISS 
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI 
# from langchain.chains.question_answering import load_qa_chain 
 
# # Load environment variables 
# load_dotenv() 
# api_key = os.getenv("GOOGLE_API_KEY") 
 
# st.set_page_config(page_title="PDF QA with Google Gemini", layout="wide") 
# st.title("📄 AI-Powered Q&A Assistant") 
 
# # Sidebar settings 
# st.sidebar.header("Settings") 
# uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf") 
# show_text_preview = st.sidebar.checkbox("Show extracted text", value=False) 
# chunk_size = st.sidebar.number_input("Chunk size", min_value=100, max_value=2000, value=800, step=100) 
# chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50) 
 
# if uploaded_file: 
#     with st.spinner("Reading PDF..."): 
#         pdf = PdfReader(uploaded_file) 
#         st.write(f"Number of pages: {len(pdf.pages)}") 
 
#         # Extract text 
#         raw_text = '' 
#         for page in pdf.pages: 
#             text = page.extract_text() 
#             if text: 
#                 raw_text += text 
         
#         st.write(f"Extracted {len(raw_text)} characters from PDF.") 
#         if show_text_preview: 
#             st.text_area("PDF Text Preview", raw_text, height=300) 
 
#     # Split text into chunks 
#     with st.spinner("Splitting text into chunks..."): 
#         text_splitter = CharacterTextSplitter( 
#             separator=".", 
#             chunk_size=chunk_size, 
#             chunk_overlap=chunk_overlap, 
#             length_function=len 
#         ) 
#         texts = text_splitter.split_text(raw_text) 
#         st.write(f"Split into {len(texts)} text chunks.") 
 
#     # Create embeddings 
#     with st.spinner("Creating embeddings..."): 
#         embeddings = GoogleGenerativeAIEmbeddings( 
#             model="models/gemini-embedding-001", google_api_key=api_key 
#         ) 
#         docsearch = FAISS.from_texts(texts, embeddings) 
#         st.success("✅ Embeddings created!") 
 
#     # Question input 
#     query = st.text_input("Ask a question about the PDF:") 
 
#     if query: 
#         with st.spinner("Generating answer..."): 
#             model = GoogleGenerativeAI( 
#                 model="gemini-2.5-flash", 
#                 google_api_key=api_key 
#             ) 
#             chain = load_qa_chain(model, chain_type="stuff") 
            
#             # Always fetch top 3 relevant chunks internally (no user control)
#             docs = docsearch.similarity_search(query, k=3) 
#             answer = chain.run(input_documents=docs, question=query) 
 
#             st.subheader("Answer:") 
#             st.write(answer) 
 
#         st.success("✅ Answer generated!")  

        

import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Chat with Your PDF 🤖", layout="wide")
st.title("📄 AI-Powered Q&A Assistant")

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("📄 Upload a PDF", type="pdf")
show_text_preview = st.sidebar.checkbox("Show extracted text", value=False)
chunk_size = st.sidebar.number_input("Chunk size", min_value=100, max_value=2000, value=800, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=200, step=50)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful AI assistant that answers questions about uploaded PDFs.")
    ]

if uploaded_file:
    with st.spinner("📚 Reading and processing your PDF..."):
        pdf = PdfReader(uploaded_file)
        raw_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        if show_text_preview:
            st.text_area("PDF Text Preview", raw_text, height=300)

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=api_key
        )
        docsearch = FAISS.from_texts(texts, embeddings)
        st.success("✅ PDF processed successfully! You can now start chatting below 👇")

    # Display conversation history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Chat input box
    user_query = st.chat_input("Ask a question about your PDF...")
    if user_query:
        # Display and save human message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append(HumanMessage(content=user_query))

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                # Gemini model and chain setup
                model = GoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key
                )
                chain = load_qa_chain(model, chain_type="stuff")

                # Retrieve relevant text chunks
                docs = docsearch.similarity_search(user_query, k=3)

                # Combine conversation context (optional, could use memory wrapper too)
                context_messages = [msg.content for msg in st.session_state.messages if isinstance(msg, HumanMessage)]
                full_prompt = "\n\n".join(context_messages[-3:])  # last 3 interactions for context

                answer = chain.run(input_documents=docs, question=user_query + "\n\nContext:\n" + full_prompt)

                st.markdown(answer)
                st.session_state.messages.append(AIMessage(content=answer))
else:
    st.info("👆 Upload a PDF file from the sidebar to start chatting.")
