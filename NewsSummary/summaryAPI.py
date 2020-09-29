from flask import Flask, request
from scrape_summarize import scrapeData, dataClean, textSummary, outputSummary
from datetime import datetime

app = Flask(__name__)

@app.route('/summary', methods=['POST'])
def summarize():
    response = request.json
    search_term = response['search_term']
    num_links = response['num_links']
    sent_count = response['sent_count']
    summary = outputSummary(search_term,num_links,sent_count)
    return summary
    

if __name__ == "__main__":
    app.run(debug=False)
