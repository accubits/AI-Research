from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options  
import regex as re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import logging
import numpy
from newspaper import Article
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

chrome_options = Options()
chrome_options.add_argument("--headless") 

def scrapeData(search_term,num_links):
    driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver",options=chrome_options)
    URL = 'https://news.google.com/search?q={}'.format(search_term)
    driver.get(URL)
    n_fail_links = 0
    links = []
    articles = ''
    for i in driver.find_elements_by_tag_name('article')[:num_links]:
        links.append(i.find_element_by_tag_name('a').get_attribute('href'))
    driver.close()
    for link in tqdm(links):
        article = Article(link)
        try:
            article.download()
            article.parse()
            articles += article.text
        except:
            n_fail_links+=1
            # print('{} is not responding'.format(driver.current_url))
    return articles,n_fail_links

def dataClean(data):
    BOW = ['[rR]ead [mM]ore.*','\[.*\]']
    for pattern in BOW:
        data = (re.sub(pattern, '', data))
    return data

def textSummary(data, SENTENCES_COUNT):
    LANGUAGE = "english"
    parser = PlaintextParser.from_string(data, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    x = ''
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        x+= ' {}'.format(str(sentence))
    return x

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--search", required=True, help='The search term')
    parser.add_argument("--num_links", default=10, help='The number of links to scrape from. default is 5')
    parser.add_argument("--sent_count", default=20, help='The number of output sentence')
    args = parser.parse_args()
    
    search_term = args.search
    num_links = args.num_links
    sent_count = args.sent_count

    data,n_fail_links = scrapeData(search_term, num_links)
    clean_data = dataClean(data)
    summary = textSummary(clean_data, sent_count)
    prefix = 'Search term: {}\nDate Created: {}\nLinks failed: {}\n\n'.format(search_term,datetime.now().strftime("%d/%m/%Y %H:%M:%S"),n_fail_links)
    summary = prefix+summary

    f = open(re.sub(' ', '_', 'outputs/summary_{}.txt'.format(search_term)), 'w')
    f.write(summary)
    f.close()