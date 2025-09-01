# main.py
from fastapi import FastAPI
from three_api_and_model_api import app as tf_app
#from three_nlp_helpers import appTwo as transformer_app

app = FastAPI()
app.mount("", tf_app)
#app.mount("/transformer", transformer_app)
