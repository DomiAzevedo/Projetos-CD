import pandas as pd
from flask import Flask, render_template, request, send_from_directory
import numpy as np

app = Flask(__name__)

# Carregar dados dos livros
try:
    books_data = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
except Exception as e:
    print(f"Failed to load CSV data:\n{e}")
    exit()
    
# Criar função para recomendar livros
def recommend_books(book):
    # Calcular relevância de cada livro
    books_data['relevance'] = books_data['average_rating'] * books_data['ratings_count']

    # Filtrar livros similares
    similar_books = books_data[(books_data['authors'] == book['authors'].values[0]) | (books_data['categories'] == book['categories'].values[0])]
    similar_books = similar_books[similar_books['title'] != book['title'].values[0]]

    # Calcula NDCG score
    similar_books['ndcg_score'] = similar_books['relevance'].rank(ascending=False, method='max') / np.log2(similar_books.index + 2)

    # Ranking dos livros e escolha do 5 melhores
    recommended_books = similar_books.sort_values(by='ndcg_score', ascending=False).head(5)

    return recommended_books

# Função para buscar livro
def search_book(book_name):
    book_name = book_name.strip().lower()
    book = books_data.loc[books_data['title'].str.lower() == book_name]
    if not book.empty:
        book_info = f"Category: {book['categories'].values[0]}\n" \
                    f"Year: {int(book['published_year'].values[0])}\n" \
                    f"Rating: {book['average_rating'].values[0]}\n" \
                    f"Pages: {int(book['num_pages'].values[0])}\n" \
                    f"Description: {book['description'].values[0]}"
        image_url = book['thumbnail'].values[0]
        title = f"{book['title'].values[0]} - {book['authors'].values[0]}"
        recommended_books = recommend_books(book)
        if not recommended_books.empty:
            list_books = []
            for index, row in recommended_books.iterrows():
                list_books.append([row['title'], row['thumbnail'], row['authors']])
            recommended_books = list_books
        else:
            recommended_books = []
    else:
        book_info = "Book not found."
        image_url = ""
        title = ""
        recommended_books = []

    return title, book_info, image_url, recommended_books

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route("/")
def index():
    return render_template("index.html", title="", book_info="", image_url="", recommended_books="")

@app.route("/search", methods=["POST"])
def search():
    book_name = request.form["book_name"]
    title, book_info, image_url, recommended_books = search_book(book_name)
    return render_template("index.html", title=title, book_info=book_info, 
                           image_url=image_url, recommended_books=recommended_books)

if __name__ == "__main__":
    app.run(debug=True)