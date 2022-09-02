from selenium import webdriver
import selenium
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import time

driver = webdriver.Chrome('./chromedriver')
url = 'https://search.shopping.naver.com/catalog/31706560655?query=%EB%84%A4%EC%9D%B4%EB%B2%84%20%EC%87%BC%ED%95%91&NaPm=ct%3Dl3kemjlk%7Cci%3D43ced300bcec6ea6696eb62373f2e6eff35da450%7Ctr%3Dslsl%7Csn%3D95694%7Chk%3Df2b46ca96bb02ee6159bc3aac6a266e601ada8cc'
driver.get(url)
time.sleep(3)

last_page_height = driver.execute_script('return document.documentElement.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0, document.documentElement.scrollHeight);')
    time.sleep(3.0)
    new_page_height = driver.execute_script('return document.documentElement.scrollHeight')

    if new_page_height == last_page_height:
                break
    last_page_height = new_page_height

comment = driver.find_elements_by_css_selector('.reviewItems_text__XIsTc') #목록 .basicList_link__1MaTN
print(len(comment))
result = []
for i in range(len(comment)):
    temp = str(comment[i].text)
    result.append(temp)
print(comment)
