from flask import Flask,render_template,request 
import pickle
import re
import nltk

nltk.download('stopwords')
nltk.download('snowball_data')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
###Loading model and cv
loaded_cv = pickle.load(open("cv.pkl", "rb"))
loaded_clf = pickle.load(open("clf.pkl", "rb"))
loaded_lb = pickle.load(open("lb.pkl", "rb"))


app = Flask(__name__) ## defining flask name

@app.route('/') ## home route
def home():
    return render_template('index.html') ##at home route returning index.html to show

@app.route('/predict',methods=['POST']) ## on post request /predict 
def predict():
    if request.method=='POST':     
        new_review = request.form['message']  ## requesting the content of the text field ## converting text into a list
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        snow = SnowballStemmer('english')
        sw = stopwords.words('english')
        new_review = [snow.stem(word) for word in new_review if not word in set(sw)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = loaded_cv.transform(new_corpus).toarray()
        new_y_pred = loaded_clf.predict(new_X_test)
        
        arr = loaded_lb.classes_
        arr[new_y_pred]
        
        return render_template('result.html',prediction=new_y_pred) ## returning result.html with prediction var value as class value(0,1)
if __name__ == "__main__":
    app.run(debug=True)     ## running the flask app as debug==True