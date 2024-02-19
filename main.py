from flask import Flask
from nltk.tokenize import wordpunct_tokenize
from recommendations.recommendations import RecommendationsEngine

app = Flask(__name__)
recsys = RecommendationsEngine('recommendations/full_data.csv', 'search_term', 'occurrences',
                               'product_title', 'relevance', wordpunct_tokenize, 0.85)


@app.route('/')
def index():
    while True:
        query = input("Search: ")
        print(recsys.get_recommendations_and_brands(query, topn=5))


if __name__ == '__main__':
    settings = {'host': '127.0.0.1', 'port': 5000}
    app.run(host=settings['host'], port=settings['port'], debug=True)
