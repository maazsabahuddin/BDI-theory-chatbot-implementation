# For any query, contact maazsabahuddin@gmail.com
from flask import Flask
from User.db import initialize_db
import os
from Main.settings.config import SECRET_KEY

MONGO_DB_PW = os.environ['MONGO_DB_PW']

from User.blueprint import my_view
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['JSON_SORT_KEYS'] = False
app.register_blueprint(my_view)

DB_URI = "mongodb+srv://maaz:"+MONGO_DB_PW+"@learn-t5bnk.mongodb.net/Assistant?retryWrites=true&w=majority"
app.config['MONGODB_HOST'] = DB_URI

initialize_db(app)


if __name__ == "__main__":
    app.run(debug=True)
