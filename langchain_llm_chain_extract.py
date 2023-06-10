import re
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import requests
from xml.etree import ElementTree
import sys
import os
import pandas as pd
from collections import Counter
from pathlib import Path

model = 'gpt-3.5-turbo'
accepted_categories = ['politics', 'environment', 'society', 'sports', 'lifestyle', 'technology', 'arts']
target_folder = Path('./llm_chain_out')

if not target_folder.exists():
    target_folder.mkdir(parents=True)

def extract_rss(url):
    """
    Extracts the content and title from a URL.
    :param url The RSS feed URL, like e.g: http://www.theguardian.com/profile/georgemonbiot/rs
    """
    response = requests.get(url)
    tree = ElementTree.fromstring(response.content)
    content = []
    for child in tree:
        if child.tag == 'channel':
            for channel_child in child:
                if channel_child.tag == 'item':
                    content.append({'content': channel_child[2].text, 'title': channel_child[0].text})
    return content


def process_llm(input_list: list, prompt_template):
    """
    Creates the LLMChain object using a specific model
    :param input_list a list of dictionaries with the content and title of each article
    :param prompt_template A single prompt template with content and title parameters
    """
    llm = ChatOpenAI(temperature=0, model=model)
    # llm = OpenAI(temperature=0, model='text-davinci-003')
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    return llm_chain.apply(input_list)


def categorize_sentiment(text):
    text = text.lower()
    if 'very negative' in text:
        return 'very negative'
    elif 'negative' in text:
        return 'negative'
    elif 'very positive' in text:
        return 'very positive'
    elif 'positive' in text:
        return 'positive'
    return 'neutral'


def sanitize_categories(text):
    text = text.lower()
    sanitized = []
    for cat in accepted_categories:
        if cat in text:
            sanitized.append(cat)
    return sanitized


def sanitize_keywords(text):
    text = text.lower()
    text = text.replace("keywords:", "").strip()
    sanitized = [re.sub(r"\.$", "", s.strip()) for s in text.split(",")]
    return sanitized


prompt_templates = [("Please tell me the sentiment of {content} with this the title: {title}? Is it very positive, positive, very negative, negative or neutral? " 
                       + "Please answer using these expressions: 'very positive', 'positive', 'very negative', 'negative' or 'neutral'"),
                       "Please extract the most relevant keywords from {content} the title: {title}. Use a the prefix 'Keywords:' before the list of keywords.",
                       "Please categorize the following content using the following content {content} with title {title} using these categories: " + ",".join(accepted_categories)]

def serialize_results(url, result_df, title, sentiment_counter: Counter, categories_counter: Counter, keywords_counter: Counter):
    """
    Converts the results to an Excel sheet or HTML page. The HTML page also contains the counter information.
    :param url The RSS feed URL
    :param result_df The combined raw data and with the LLM output
    :param title The RSS feed URL with some modified characters
    :param sentiment_counter The counter with the sentiment information
    :param categories_counter The counter with the counted categories
    """
    result_df.to_excel(target_folder/f"{title}.xlsx")
    html_file = target_folder/f"{title}.html"
    html_content = result_df.to_html(escape=False)
    # Make sure the file is written in UTF-8
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(html_content)
    sentiment_html = generate_sentiment_table(sentiment_counter, "Sentiment")
    categories_html = generate_sentiment_table(categories_counter, "Category")
    keywords_html = generate_sentiment_table(dict(keywords_counter.most_common()[:10]), "Keywords")
    with open(html_file, encoding="utf8") as f:
        content = f"""<html>
                <head>
                    <meta charset="UTF-8" />
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>
                </head>
                <body>
                    <div class="container-fluid">
                        <h1>{re.sub(r'.+?theguardian.com/profile', '', url).replace("/rss", "").replace("/", "")}</h1>
                        <h3>Sentiment Count</h3>
                        {sentiment_html}
                        <h4>Categories Count</h4>
                        {categories_html}
                        <h4>Keywords Count</h4>
                        {keywords_html}
                        {f.read()}
                    </div>
                </body>
            </html>"""
        content = content.replace('class="dataframe"', 'class="table table-striped table-hover dataframe"')
    with open(html_file, "w", encoding="utf8") as f:
        f.write(content)

def generate_sentiment_table(sentiment_counter, title):
    sentiment_html = f"<table class='table table-hover'><tr><th>{title}</th><th>Count</th></tr>"
    for s in sentiment_counter:
        sentiment_html += f"<tr><td style='max-width: 200px; width: 100px'>{s}</td><td>{sentiment_counter[s]}</td></tr>"
    sentiment_html += "</table>"
    return sentiment_html

def process_url(url):
    """
    Extracts the content of each RSS Feed.
    Sends the content of each RSS feed to the LLMChain to apply the prompts to the extracted records.
    Creates a data set for each RSS feed which combines the output of the LLM and generates an HTML and Excel file out of it.
    :param url: the URL of the RSS feed, like e.g: http://www.theguardian.com/profile/georgemonbiot/rss
    """
    print(f"Processing {url}")
    zipped_results = []
    llm_responses = []
    input_list = extract_rss(url)
    for prompt_template in prompt_templates:
        llm_responses.append(process_llm(input_list, prompt_template))
    sentiment_counter = Counter()
    categories_counter = Counter()
    keywords_counter = Counter()
    for zipped in zip(input_list, *llm_responses):
        sentiment = {'sentiment': zipped[1]['text']}
        categorized_sentiment = categorize_sentiment(zipped[1]['text'])
        sentiment_counter[categorized_sentiment] += 1
        sentiment_category = {'sentiment_category': categorized_sentiment}
        keywords = {'keywords': zipped[2]['text']}
        raw_categories = zipped[3]['text']
        classification = {'classification': raw_categories}
        sanitized_topics = sanitize_categories(raw_categories)
        categories_counter.update(sanitized_topics)
        raw_keywords = zipped[2]['text']
        keywords_counter.update(sanitize_keywords(raw_keywords))
        sanitized_categories = {'topics': ",".join(sanitized_topics)}
        full_record = {
            **zipped[0], 
            **sentiment, 
            **keywords, 
            **sentiment_category, 
            **classification,
            **sanitized_categories
        }
        zipped_results.append(full_record)
    result_df = pd.DataFrame(zipped_results)
    title = url.replace(":", "_").replace("/", "_")
    serialize_results(url, result_df, title, sentiment_counter, categories_counter, keywords_counter)

if __name__ == "__main__":
    # Example:
    # python .\langchain_llm_chain_extract.py http://www.theguardian.com/profile/georgemonbiot/rss http://www.theguardian.com/profile/simonjenkins/rss 
    # http://www.theguardian.com/profile/zoewilliams/rss http://www.theguardian.com/profile/marinahyde/rss http://www.theguardian.com/profile/pollytoynbee/rss https://www.theguardian.com/profile/owen-jones/rss
    # https://www.theguardian.com/profile/jonathanfreedland/rss https://www.theguardian.com/profile/johncrace/rss
    # Configuration:
    # Do not forget to set OPENAI_API_KEY in your environment
    # os.environ["OPENAI_API_KEY"] = '<key>'
    if len(sys.argv) == 1:
        print("Please enter the URLs from which the titles are to be extracted.")
        sys.exit()
    input_list = []
    for url in sys.argv[1:]:
        process_url(url)
            


        
        


