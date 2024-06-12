import pandas as pd
from flask import Flask, render_template, request, send_from_directory
import numpy as np

app_flask = Flask(__name__)
from projeto_atualizado.book_rec_app import VespaApp

app_vespa = VespaApp()

empty_df = pd.DataFrame(columns=["id", "title", "authors", "description", "categories", "thumbnail"])

@app_flask.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app_flask.route("/")
def index():
    return render_template("index.html", recommended_books=empty_df)

@app_flask.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    recommended_books = app_vespa.query_bm25(query)
    df = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
    recommended_books = recommended_books.merge(df[['title', 'thumbnail']], on='title', how='left')
    return render_template("index.html", recommended_books=recommended_books)

if __name__ == "__main__":
    app_flask.run(debug=True)