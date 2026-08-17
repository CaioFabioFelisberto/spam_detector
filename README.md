# 📧 Detector de Spam com Naive Bayes & Flask

Projeto de Machine Learning e Processamento de Linguagem Natural (PLN) para identificação automática de mensagens de spam. O modelo foi treinado com o algoritmo **Multinomial Naive Bayes** e disponibilizado através de uma **API REST/Interface Web** construída com **Flask**.

---

## ⚠️ Nota Importante sobre os Testes

> **O modelo deve ser testado com textos em INGLÊS.**
> 
> O conjunto de dados utilizado no treinamento (*SMS Spam Collection*) contém exclusivamente mensagens na língua inglesa. Caso você digite frases em português, as palavras serão ignoradas pelo vetorizador (por estarem fora do vocabulário de treino) e o modelo classificará a mensagem com a probabilidade padrão (*Ham*).

### Exemplos para teste:
* **Spam:** `"WINNER!! Claim your free $1000 prize right now! Call 08002986030"`
* **Ham (Legítimo):** `"Hey mom, I will be home around 7 pm for dinner."`

---

## 🚀 Funcionalidades

- **Processamento de Texto:** Conversão de texto bruto em vetores numéricos usando **TF-IDF**.
- **Classificação Probabilística:** Algoritmo Multinomial Naive Bayes calculando a probabilidade de a mensagem ser Spam ou Ham.
- **API REST & Web UI:** Interface gráfica e rotas para integração com outros sistemas via requisições HTTP (`POST`).
- **Persistência de Artefatos:** Modelo e vetorizador serializados com `joblib` para rápida execução em produção.

---

## 📁 Estrutura do Projeto

```text
detector_de_spam/
│
├── .venv/                      # Ambiente virtual
├── artifacts/                  # Artefatos exportados do ML
│   ├── spam_model.pkl          # Modelo Naive Bayes treinado
│   └── vectorizer.pkl          # Vetorizador TF-IDF ajustado
│
├── data/                       # Base de dados
│   └── spam.csv                # Dataset SMS Spam Collection
│
├── src/                        # Scripts de Machine Learning
│   └── machine.py              # Pré-processamento, treino e avaliação
│
├── templates/                  # Interface Web
│   └── predict.html            # Tela de testes em HTML/Bootstrap/JS
│
├── app.py                      # Servidor Flask (API e Rotas Web)
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação

# Como executar o projeto
## 1. Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado.

## 2. Clonar e Instalar Dependências
```bash
pip install -r requirements.txt
```

## 3. Treinar o Modelo (Opcional)
Caso queira refazer o treinamento e gerar novos artefatos .pkl:
```bash
python src/machine.py
```

## 4. Iniciar a Aplicação Web
```bash
python app.py
```
Acesse http://127.0.0.1:5000 no seu navegador para utilizar a interface gráfica.

# Endpoints da API
POST /predict
Recebe um texto em inglês e retorna a classificação junto com o nível de confiança.
Payload:
```JSON
{
    "texto": "WINNER!! Claim your free $1000 prize right now!"
}
```
Resposta
````JSON
{
    "classe": "spam",
    "probabilidade": 0.9854
}
```