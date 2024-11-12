

import threading
import webbrowser
from flask import Flask, request, jsonify, render_template
import nltk  
from nltk.sentiment import SentimentIntensityAnalyzer  
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration  
from transformers import AutoModelForCausalLM, AutoTokenizer
app = Flask(__name__)  

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

nltk.download('vader_lexicon')  

def generate_response(user_input):  
    inputs = tokenizer(user_input, return_tensors='pt')  
    response_id = model.generate(**inputs)  
    response = tokenizer.batch_decode(response_id, skip_special_tokens=True)[0]  
    return response

def run_app():
    app.run(debug=True, use_reloader=False)
def detect_emotion(text):  
    sia = SentimentIntensityAnalyzer()  
    sentiment_scores = sia.polarity_scores(text)  
    if sentiment_scores['compound'] >= 0.05:  
        emotion = "Positive"  
    elif sentiment_scores['compound'] <= -0.05:  
        emotion = "Negative"  
    else:  
        emotion = "Neutral"
    return emotion

@app.route('/')  
def home():  
    return render_template('index.html')  # Serve the HTML page  

@app.route('/conversation', methods=['POST'])  
def conversation():  
    user_input = request.json.get('message')  
    response = generate_response(user_input)  
    emotion = detect_emotion(user_input)  # Detect emotion based on user input
    return jsonify({'response': response, 'emotion': emotion})

if __name__ == '__main__':
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    run_app()