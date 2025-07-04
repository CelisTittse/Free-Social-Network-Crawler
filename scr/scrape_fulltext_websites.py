import scrapy
import json
from w3lib.html import remove_tags, remove_comments
from random import choices
from bs4 import BeautifulSoup
import html

gt_json_path = '/Users/celistittse/Documents/CultTech/Data/raw_data/gt_websites.json'

# get clean text from html
def cleaner_text (raw_html: str) -> str:
    # Step 1: Parse and strip HTML tags
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n")  # Get visible text with newlines

    # Step 2: Unescape HTML entities
    text = html.unescape(text)

    # Step 3: Clean up (remove excessive blank lines)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_text = " ".join([line for line in lines if line])

    # replace enter with point space and remove double enters
    cleaned_text = cleaned_text.replace('\n', '. ').replace('  ', ' ')
    cleaned_text = cleaned_text.replace('\t', ', ').replace('  ', ' ')

    return cleaned_text


# open json
with open(gt_json_path, encoding='utf-8-sig') as gt_f:
    gt_json = json.load(gt_f)
    sample_keys = choices(list(gt_json.keys()), k=3)
    gt_test = {k: v for k, v in gt_json.items() if k in sample_keys}


class FullTextSpider(scrapy.Spider):
    name = "full_text"

    # use Ground Truth websites with id.{0,1} as keys
    input_dict = gt_json

    def start_requests(self):
        for key, urls in self.input_dict.items():
            for url in urls:
                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                    meta={"source_key": key}
                )

    def parse(self, response):
        source_key = response.meta["source_key"]
        raw_html = response.text
        clean_text = cleaner_text(raw_html=raw_html)

        yield {
            "key": source_key,
            "url": response.url,
            "full_text": clean_text.strip()
        }
