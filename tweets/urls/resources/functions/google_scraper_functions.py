# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:31:22 2023

@author: Shiyi Shen
"""

from googlesearch import search
import requests
from lxml import html
import pandas as pd
from time import sleep
from typing import List, Tuple

def goog_search(term, num_search, pausing) -> Tuple[str]:
    try:
        # print the string we're searching
        print(term)
        # for more details on this function check out
        # https://pypi.org/project/googlesearch-python/
        search_list = search(term, num_results=num_search)
        sleep(pausing)

        def populate(urls):
            for i in urls:
                print(i)
                yield i
        return tuple(populate(search_list))

    except:
        return ['']


def clean_camp_sites(row) -> List[str]:
    # clean up the sites and url and stuff
    try:
        names_lower = row['CAND_NAME'].lower().split()
        # remove middle initials
        names_lower = [x for x in names_lower if len(x) > 1]
        # remove any special characters
        names_lower = [x.replace(',', '') for x in names_lower]
        href_list = row['campaign_results_list']
        href_list = ['/'.join(x.lower().split('/')[:4]) for x in href_list]

        fill_array = []
        for x in href_list:

            if any(n in x for n in names_lower):
                fill_array.append(x)

        return fill_array
    except:
        return ['']


def try_get_url(tree_in, text) -> str:
    try:
        a = tree_in.xpath("//a[contains(text(),'" + text + "')]")
        a = [x.attrib['href'] for x in a][0]
    except:
        a = ''
    return a


def get_links_from_ballot(urls) -> pd.Series:
    try:
        page = requests.get(list(urls)[0])
        sleep(3)
        print(page.status_code)
        tree = html.fromstring(page.content)

        campaign_fb = try_get_url(tree, 'Campaign Facebook')
        campaign_site = try_get_url(tree, 'Campaign website')
        campaign_twitter = try_get_url(tree, 'Campaign Twitter')

        personal_fb = try_get_url(tree, 'Personal Facebook')
        personal_linkedin = try_get_url(tree, 'Personal LinkedIn')
        personal_twitter = try_get_url(tree, 'Personal Twitter')

        official_fb = try_get_url(tree, 'Official Facebook')
        official_site = try_get_url(tree, 'Official website')
        official_twitter = try_get_url(tree, 'Official Twitter')

        return [campaign_fb, campaign_site, campaign_twitter,
                          personal_fb, personal_linkedin, personal_twitter,
                          official_fb, official_site, official_twitter]
    except:
        return ['']
