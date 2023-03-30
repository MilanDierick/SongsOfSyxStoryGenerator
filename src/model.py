import openai


class Model:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", system_content: str = ""):
        self.api_key = api_key
        self.model = model
        self.system_content = system_content
        openai.api_key = self.api_key

    def get_api_key(self):
        return self.api_key

    def get_model(self):
        return self.model

    def get_system_content(self):
        return self.system_content

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def set_model(self, model: str):
        self.model = model

    def set_system_content(self, system_content: str):
        self.system_content = system_content

    def get_response(self, prompt: str, stream: bool = False):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_content},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            stream=stream
        )

        if stream:
            return iter(response)
        else:
            return response

    def print_response(self, prompt: str):
        response = self.get_response(prompt, stream=True)

        for chunk in response:
            # handle multiple choices
            for choice in chunk['choices']:
                dictionary: dict = choice['delta']
                if 'content' in dictionary:
                    print(dictionary['content'], end='')
