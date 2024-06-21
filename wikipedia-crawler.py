#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import argparse
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from torch import cosine_similarity
import urllib3
import json
import httpx
from openai import OpenAI

# Suppress only the single InsecureRequestWarning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_OUTPUT = 'output.txt'
DEFAULT_INTERVAL = 5.0  # interval between requests (seconds)
DEFAULT_ARTICLES_LIMIT = 1  # total number articles to be extrated
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'


pending_urls = []  # queue


def load_urls(session_file):
    """Resume previous session if any, load visited URLs"""
    visited_urls = set()  # all urls already visited, to not visit twice
    try:
        with open(session_file) as fin:
            for line in fin:
                visited_urls.add(line.strip())
    except FileNotFoundError:
        pass
    
    return visited_urls
    
import re

MANUFACTURING_KEYWORDS = [
    'manufacturing', 'industrial', 'production', 'factory', 'machinery',
    'automation', 'assembly', 'fabrication', 'process', 'engineering'
]

# Calculate cosine similarity
from scipy.spatial.distance import cosine
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def is_relevant(anchor_embeddings, text, embedding_model, openai_client, required_similarity):
    embedding = get_embedding_by_text(text, embedding_model, openai_client)
    max_similarity = 0
    for anchor_embedding in anchor_embeddings:
        #convert the embeddings to pytorch tensors
        
        similarity = cosine_similarity(embedding, anchor_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
        if similarity > required_similarity:
            return True, max_similarity
    return False, max_similarity


def scrap(base_url, article, max_tokens_per_article=8192):
    """Represents one request per article"""

    full_url = base_url + article
    r = requests.get(full_url, headers={'User-Agent': USER_AGENT}, verify=False)
    
    if r.status_code not in (200, 404):
        print("Failed to request page (code {})".format(r.status_code))
        time.sleep(10)
        raise Exception("Failed to request page (code {})".format(r.status_code))

    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find('div', {'id':'mw-content-text'})

    # add new related articles to queue
    # check if are actual articles URL
    hrefs = []
    for a in content.find_all('a'):
        href = a.get('href')
        if not href:
            continue
        if href[0:6] != '/wiki/':  # allow only article pages
            continue
        elif ':' in href:  # ignore special articles e.g. 'Special:'
            continue
        elif href[-4:] in ".png .jpg .jpeg .svg":  # ignore image files inside articles
            continue
        hrefs.append(href)

    parenthesis_regex = re.compile('\(.+?\)')  # to remove parenthesis content
    citations_regex = re.compile('\[.+?\]')  # to remove citations, e.g. [1]

    # get plain text from each <p>
    p_list = content.find_all('p')
    all_text = ''
    with open(output_file, 'a', encoding='utf-8') as fout:
        for p in p_list:
            text = p.get_text().strip()
            text = parenthesis_regex.sub('', text)
            text = citations_regex.sub('', text)
            lines = text.splitlines()
            
            # Check if any line in the paragraph is shorter than 10 characters
            if any(len(line.strip()) < 10 for line in lines):
                continue  # Skip paragraphs with short lines
            
            if text.endswith('.'):
                if num_tokens(all_text + text) > max_tokens_per_article:
                    break
                all_text += text + '\n'
                
    return all_text, hrefs

import tiktoken

def num_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return len(encoding.encode(text))


def get_embedding_by_url(url, embedding_model, openai_client):
    # fetch the text from the given url
    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(url))
    article = url[len(base_url):]
    all_text, _ = scrap(base_url, article)
    response = openai_client.embeddings.create(
        input=all_text,
        model=embedding_model
    )
    return response.data[0].embedding

def get_embedding_by_text(text, embedding_model, openai_client):
    # retry 3 times
    retry = 0
    max_retry = 3
    while retry < max_retry:
        try:
            response = openai_client.embeddings.create(
                input=text,
                model=embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(1)
            retry += 1
            continue


def get_anchor_embeddings(anchors, embedding_model, openai_client):
    anchor_embeddings = []
    for anchor in anchors:
        print(f"Getting embeddings for anchor: {anchor}")
        # Get embeddings for anchor
        anchor_embedding = get_embedding_by_url(anchor, embedding_model, openai_client)
        anchor_embeddings.append(anchor_embedding)
    return anchor_embeddings


def main(anchors, embedding_model, openai_client, articles, interval, output_file, max_tokens_per_article, required_similarity):
    """ Main loop, single thread """

    minutes_estimate = interval * articles / 60
    print("This session will take {:.1f} minute(s) to download {} article(s):".format(minutes_estimate, articles))
    print("\t(Press CTRL+C to pause)\n")
    session_file = "session_" + output_file
    pending_file = "pending_" + output_file
    visited_urls = load_urls(session_file)  # load previous session (if any)
    pending_urls = list(load_urls(pending_file))
    
    anchor_embeddings = get_anchor_embeddings(anchors, embedding_model, openai_client)
    
    
    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(anchors[0]))
    
    # first time:
    if len(pending_urls) == 0 and len(visited_urls) == 0:
        for anchor in anchors:
            article = anchor[len(base_url):]
            pending_urls.append(article)
            
            
    print('pending_urls', pending_urls)
    counter = 0
    while len(pending_urls) > 0:
        try:
            counter += 1
            if counter > articles:
                break
            try:
                next_url = pending_urls.pop(0)
            except IndexError:
                break
            
            
            article_format = next_url.replace('/wiki/', '')[:35]
            print("{:<7} {}".format(counter, article_format))
            retry = 0
            while retry < 3:
                try:
                    text, hrefs = scrap(base_url, next_url, max_tokens_per_article)
                    if text is not None:
                        break
                except:
                    print(f"Failed to scrap article {next_url}, retrying...")
                    time.sleep(10)
                    retry += 1
                    continue
                    
            if text is None:
                print(f"Skipping article {full_url} as it has no text.")
                continue
            
            full_url = base_url + next_url
            if full_url not in visited_urls:
                relevant, max_similarity = is_relevant(anchor_embeddings, text, embedding_model, openai_client, required_similarity)
                if not relevant:
                    print(f"Skipping article {full_url} as {max_similarity} is less than the required similarity: {required_similarity}.")
                    continue
            
                # skip if already added text from this article, as continuing session
                with open(output_file, 'a', encoding='utf-8') as fout:
                    fout.write(text)
                    fout.write('\n\n')
                    
                with open(session_file, 'a') as fout:
                    fout.write(full_url + '\n')
                
                visited_urls.add(full_url)

            
            for href in hrefs:
                if href not in pending_urls and href not in visited_urls and len(pending_urls) < 1000:
                    pending_urls.append(href)
                    
            # overwrite pending files every 10 loops
            if counter % 10 == 0:
                with open(pending_file, 'w') as fout:
                    for url in pending_urls:
                        fout.write(url + '\n')
                        
            time.sleep(interval)
            
        except KeyboardInterrupt:
            input("\n> PAUSED. Press [ENTER] to continue...\n")
            counter -= 1

    print("Finished!")
    sys.exit(0)


if __name__ == '__main__':
    # read file 'config.json'
    with open('config.json') as json_file:
        data = json.load(json_file)
        anchors = data['anchors']
        embedding_model = data['embedding-model']
        api_key = data['api-key']
        articles = data['articles']
        interval = data['interval']
        output_file = data['output-file']
        max_tokens_per_article = data['max-tokens-per-article']
        required_similarity = data['required-similarity']
    http_client = httpx.Client(verify=False)
    openai_client = OpenAI(api_key=api_key,  http_client=http_client)
    main(anchors, embedding_model, openai_client, articles, interval, output_file, max_tokens_per_article, required_similarity)


