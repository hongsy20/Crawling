from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
import urllib3


#search = input('검색어를 입력하세요: ')
url = 'https://www.youtube.com/watch?v=muR_S-4x3Qc'
driver = webdriver.Chrome("./chromedriver")
driver.get(url)

last_page_height = driver.execute_script('return document.documentElement.scrollHeight')

#화면을 맨 밑에까지 내려줌
while True:
    driver.execute_script('window.scrollTo(0, document.documentElement.scrollHeight);')
    time.sleep(3.0)
    new_page_height = driver.execute_script('return document.documentElement.scrollHeight')

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height
#####################################################################
html_source = driver.page_source

driver.close()

soup = BeautifulSoup(html_source, 'lxml')
#####################################################################

#유튜브 댓글 아이디, 댓글
youtube_user_id = soup.select('#author-text > span')
youtube_user_comment = soup.select('#content-text')
#####################################################################

#위에서 받아온 내용 중 의미없는 값 처리
str_youtube_id = []
str_youtube_comment = []

for i in range(len(youtube_user_id)):
    str_tmp = str(youtube_user_id[i].text)
    
    str_tmp = str_tmp.replace('\n', '')
    str_tmp = str_tmp.replace('\t', '')
    str_tmp = str_tmp.replace('              ', '')
    str_youtube_id.append(str_tmp)

    str_tmp = str(youtube_user_comment[i].text)
    str_tmp = str_tmp.replace('\n', '')
    str_tmp = str_tmp.replace('\t', '')
    str_tmp = str_tmp.replace('             ', '')
    str_youtube_comment.append(str_tmp)

import pandas as pd

pd_data = {'ID':str_youtube_id, 'Comment':str_youtube_comment}

youtube_pd = pd.DataFrame(pd_data)
youtube_pd.to_csv('filename.csv', mode = 'w')

print('완료')
