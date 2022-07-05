from flask import Flask, request
from keras.models import load_model
import spam_detecting_service  
import nltk

app = Flask(__name__)
#app.debug = True

loaded_model=load_model("MODEL.hdf5")
nltk.data.path.append("nltk_data")

@app.route('/spam_detection', methods=['POST'])
def spam_detection():
    data = request.get_json()
    text = data['message']
    try:
        spam_meter=spam_detecting_service.spam_detector(text)
    except Exception as e :
        print(e)

    return {"spam_meter":"{:.2f} %".format(spam_meter)}

if __name__ == "__main__": 
    app.run(host="127.0.0.1", port=9856)