import openai


class Embedding:
    def __init__(self, text, guid, embedding=None, model_name='text-embedding-ada-002'):
        self.text = text
        self.guid = guid
        self.model_name = model_name

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = self.create_embedding()

    def create_embedding(self):
        result = openai.Embedding.create(
            model=self.model_name,
            input=self.text
        )

        return result['data'][0]['embedding']
