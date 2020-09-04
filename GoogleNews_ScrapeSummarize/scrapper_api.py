from flask import Flask, request
from scrape_summarize import scrapeData, dataClean, textSummary
from datetime import datetime

app = Flask(__name__)

@app.route('/summary', methods=['POST'])
def summarize():
    response = request.json
    search_term = response['search_term']
    num_links = response['num_links']
    sent_count = response['sent_count']
    articles,n_fail_links = scrapeData(search_term,num_links)
    clean_data = dataClean(articles)
    summary = textSummary(clean_data, sent_count)
    prefix = 'Search term: {}\nDate Created: {}\nLinks failed: {}\n\n'.format(search_term,datetime.now().strftime("%d/%m/%Y %H:%M:%S"),n_fail_links)
    summary = prefix+summary
    return summary
    

if __name__ == "__main__":
    app.run(debug=True)