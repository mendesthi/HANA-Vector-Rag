from flask_cors import CORS
from flask import Flask, request, jsonify

import sys, os
# Get the absolute path to the directory containing api.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory and add it to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vectorapp.modules.uploadPDF import upload_pdf_blueprint
from vectorapp.modules.uploadHTML import upload_html_blueprint
from vectorapp.modules.chunkAndStore import chunk_and_store_blueprint
from vectorapp.modules.chunkFromWebAndStore import chunk_and_store_web_blueprint
from vectorapp.modules.getContext import get_context_blueprint
from vectorapp.modules.getAnswer import get_answer_blueprint
from vectorapp.modules.getDirectAnswer import get_direct_answer_blueprint
from vectorapp.modules.embeddingFromText import embed_from_text_blueprint
from vectorapp.modules.insertTextAsVector import insert_txt_as_vector_blueprint

app = Flask(__name__)
CORS(app)

app.register_blueprint(upload_pdf_blueprint)
app.register_blueprint(upload_html_blueprint)
app.register_blueprint(chunk_and_store_blueprint)
app.register_blueprint(chunk_and_store_web_blueprint)
app.register_blueprint(get_context_blueprint)
app.register_blueprint(get_answer_blueprint)
app.register_blueprint(get_direct_answer_blueprint)
app.register_blueprint(embed_from_text_blueprint)
app.register_blueprint(insert_txt_as_vector_blueprint)

@app.route('/', methods=['GET'])
def root():
    return 'Embeddings API: Health Check Successfull.', 200

def create_app():
    return app

if __name__ == '__main__':
    app.run('0.0.0.0', 8080)