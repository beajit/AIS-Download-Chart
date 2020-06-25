
from flask import Flask

UPLOAD_FOLDER = '/home/ajitkumar/Documents/code/python/Flask/AIS/upload'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['FLASK_DEBUG']=1