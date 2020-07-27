from flask import Flask, render_template, request
import re
#import Spam_ML
import pickle
import pandas.testing as tm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)
ps = PorterStemmer()
classifier = pickle.load(open('model.pk1', 'rb'))
cv = pickle.load(open('model.pk2', 'rb'))

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route("/success", methods=['POST'])
def success():
    new_review = request.values.get("t1")
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    new_review = [ps.stem(word) for word in new_review if not word in set(stopwords.words('english'))]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    if request.method == 'POST' and new_y_pred == 0: 
        return render_template("success.html")
    elif request.method == 'POST' and new_y_pred == 1:
        return render_template("failure.html")

if __name__ == '__main__':
    app.debug=True
    app.run(use_reloader=False)