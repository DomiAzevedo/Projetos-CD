from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from book_rec_app import VespaApp

app_flask = Flask(__name__)
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
    rank_type = request.form.get("rank_type", "bm25") 

    if rank_type == "bm25":
        recommended_books = app_vespa.query_bm25(query)
    elif rank_type == "bm25_semantic":
        recommended_books = app_vespa.query_second_phase(query)
    elif rank_type == "colbert_local":
        recommended_books = app_vespa.query_colbert(query)
    elif rank_type == "colbert_global":
        recommended_books = app_vespa.query_colbert_global(query)
    elif rank_type == "hibrido_colbert_local":
        recommended_books = app_vespa.query_colbert_2phase(query)
    elif rank_type == "hibrido_colbert_global":
        recommended_books = app_vespa.query_colbert_2phase_global(query)

    df = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")

    recommended_books = recommended_books.merge(df[['title', 'thumbnail']], on='title', how='left')

    return render_template("index.html", recommended_books=recommended_books, rank_type=rank_type)

if __name__ == "__main__":
    app_flask.run(debug=True, use_reloader = False)