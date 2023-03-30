import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, Response, make_response
from database import EmbeddingDatabase
from embedding import Embedding
from model import Model

app = Flask(__name__)

# read the api key from a file
with open('api_key.txt', 'r') as f:
    key = f.read().strip()

gpt_model = Model(api_key=key)
database = EmbeddingDatabase()


def process_text(text):
    paragraphs = text.split('\n\r\n')
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        embedding = Embedding(text=paragraph, guid=uuid.uuid4())
        database.insert_embedding(embedding)


@app.route('/')
def index():
    return redirect(url_for('hello_world'))


@app.route('/index')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_text():
    if request.method == 'POST':
        text = request.form['text']
        process_text(text)
        return redirect(url_for('upload_text'))
    return render_template('upload.html')


@app.route('/embeddings')
def display_embeddings():
    embeddings: list[Embedding] = database.get_embeddings()
    return render_template('embeddings.html', embeddings=embeddings)


@app.route('/prompt', methods=['GET', 'POST'])
def prompt():
    response = None
    if request.method == 'POST':
        prompt_text = request.form['prompt']
        # Call GPT-3 API here with prompt_text
        response = gpt_model.get_response(prompt_text, stream=False)['choices'][0]['message']['content']
    return render_template('prompt.html', response=response)


@app.route('/stream_response', methods=['POST'])
def stream_response():
    prompt_text = request.form['prompt']

    def generate():
        for chunk in gpt_model.get_response(prompt_text, stream=True):
            for choice in chunk['choices']:
                dictionary: dict = choice['delta']
                if 'content' in dictionary:
                    yield dictionary['content']
            time.sleep(0.1)

    response = make_response(Response(generate(), content_type='text/html'))
    response.headers['X-Accel-Buffering'] = 'no'
    return Response(generate(), content_type='text/html')


if __name__ == '__main__':
    app.run(debug=True)
