from groq import Groq
import os
import json

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

# Model オブジェクト → dict に変換して JSON 化
models_dict = [m.model_dump() for m in models.data]

print(json.dumps(models_dict, indent=2))

