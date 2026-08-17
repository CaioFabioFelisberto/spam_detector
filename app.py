from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)


model = joblib.load("artifacts/spam_model.pkl")
vector = joblib.load("artifacts/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()
    
    if not dados or 'texto' not in dados:
        return jsonify({"erro": "Envie um JSON contendo a chave 'texto'"}), 400

    texto = dados['texto']
    
    texto_vec = vector.transform([texto])
    
    predicao = model.predict(texto_vec)[0]
    probabilidades = model.predict_proba(texto_vec)[0]
    
    prob_max = float(max(probabilidades))

    return jsonify({
        "classe": predicao,
        "probabilidade": round(prob_max, 4)
    })


if __name__ == '__main__':
    app.run(debug=True)