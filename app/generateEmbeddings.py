import os
import PyPDF2
import tempfile
import configparser

from flask_cors import CORS
from flask import Flask, request, jsonify

# Langchain to help with Text Chuncks generation
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Langchain to work with HANA Vector Engine
from langchain_community.vectorstores.hanavector import HanaDB

# Langchain to help with chat completition
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# For consuming SAP Generative AI Hub CHAT COMPLETITION model
from gen_ai_hub.proxy.langchain.init_models import init_llm
#from langchain_openai import OpenAI #REPLACED BY THE GenAI HUB SDK LANGUAGE MODEL above ;-)

# For consuming SAP Generative AI Hub EMBEDDINGS model
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
#from langchain_openai import OpenAIEmbeddings #REPLACED BY THE GenAI HUB SDK EMBEDDINGS MODEL above ;-)

# HANADB Client to initiate DB connection
from hdbcli import dbapi

app = Flask(__name__)
CORS(app)

# Check if the application is running on Cloud Foundry
if 'VCAP_APPLICATION' in os.environ:
    # Running on Cloud Foundry, use environment variables
    hanaURL = os.getenv('DB_ADDRESS')
    hanaPort = os.getenv('DB_PORT')
    hanaUser = os.getenv('DB_USER')
    hanaPW = os.getenv('DB_PASSWORD')
else:
    # Not running on Cloud Foundry, read from config.ini file
    config = configparser.ConfigParser()
    config.read('config.ini')
    hanaURL = config['database']['address']
    hanaPort = config['database']['port']
    hanaUser = config['database']['user']
    hanaPW = config['database']['password']
                     
def get_text_from_request():
    print('TCM: Getting text from request')
    request_data = request.get_json()
    return request_data['text']

def get_embeddings_from_request():
    print('TCM: Getting embeddings from request')
    request_data = request.get_json()
    return request_data['embeddings']

@app.route('/', methods=['GET'])
def root():
    return 'Embeddings API: Health Check Successfull.', 200

@app.route('/generateEmbeddings', methods=['POST'])
def upload_file():
    print('TCM: Reading the file')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Let's upload the PDF file from local user folder 
    # to our app's temporary folder
    if file:
        filename = file.filename
        pdf_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(pdf_path)
        print('TCM: PDF saved for processing')

        # Now we open the PDF file and extract the texts
        text = ''
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        
        # We create a txt file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', txt_filename)
        
        # And write the text to the text file
        with open(txt_path, 'w') as txt_file:
            print('TCM: Writting TXT file')
            txt_file.write(text)
        
        # Let's close the txt file and delete the PDF file
        print('TCM: Closing TXT file')
        txt_file.close()
        os.remove(pdf_path)
        
        # Now we call the function that will process the text file
        run_langchain_doc_load(txt_path)

        # After processing the text file, we can now delete it
        os.remove(txt_path)
        print('TCM: TXT file deleted')
        return jsonify({'value': 'Done'}), 200

@app.route('/langchain-doc-load', methods=['POST'])
def run_langchain_doc_load(filepath2load):
    # Let's load the file into a variable
    text_documents = TextLoader(filepath2load).load()

    # Then we configure the way we want to generate the chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["Q."])

    # And create the chunks
    text_chunks = text_splitter.split_documents(text_documents)

    try:
        print('TCM: Connecting to HANA Cloud DB')
        # Now we connect to the HANA Cloud instance
        conn = dbapi.connect(
            address=hanaURL,
            port=hanaPort,
            user=hanaUser,
            password=hanaPW
        )
        
        # We initialize the Embeddings model from our Generative AI Hub
        embed = init_embedding_model('text-embedding-ada-002')
        
        # And create a LangChain VectorStore interface for the HANA database and 
        # specify the table (collection) to use for accessing the vector embeddings
        db = HanaDB(
            embedding=embed, connection=conn, table_name="GENAIQA"
        )

        # For this example, we delete any previous content from
        # the table which might exist from previous runs.
        db.delete(filter={})

        # add the loaded document chunks
        response = db.add_documents(text_chunks)
        print('TCM: Chunks added to TABLE')
        
        return jsonify({'response':response}),200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/langchain-similarity', methods=['GET'])
def run_langchain_similarity():
    query = get_text_from_request() 
    try:
        #Initialize DB connection
        conn = dbapi.connect(
            address=hanaURL,
            port=hanaPort,
            user=hanaUser,
            password=hanaPW
        )
        
        # We initialize the Embeddings model from our Generative AI Hub
        embed = init_embedding_model('text-embedding-ada-002')

        # And create a LangChain VectorStore interface for the HANA database and 
        # specify the table (collection) to use for accessing the vector embeddings
        db = HanaDB(
            embedding=embed, connection=conn, table_name="GENAIQA"
        )

        # Perform a query to get the two best matching document chunks 
        # from the ones that we added in the previous step.
        # By default "Cosine Similarity" is used for the search.
        docs = db.similarity_search(query, k=2)
        
        # Extract the text from the most similar chunks
        # and assign it to the context variable
        docs_texts = []
        for doc in docs:
            docs_texts.append(doc.page_content)
        context = docs_texts[0] + ' ' + docs_texts[1]

        response = run_chat_response(query, context)
        return jsonify({'answer': response}),200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/chat-response', methods=['GET'])
def run_chat_response(promptFromUser, textFromHana):
    try:
        template = """Question: {question}
            Answer: Based on the data provided: """
        prompt = PromptTemplate(template=template, input_variables=['question'])
        question = promptFromUser + 'Context: ' + textFromHana

        llm = init_llm('gpt-4', max_tokens=150)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.invoke(question)

        print(response['text'])
        return(response['text'])
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    
def create_app():
    return app

if __name__ == '__main__':
    app.run('0.0.0.0', 8080)