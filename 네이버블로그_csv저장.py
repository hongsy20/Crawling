from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen
import csv
import urllib3

#driver = webdriver.Chrome("./chromedriver")
#driver.get(url)

#api_txt_lines total_tit

search = input('검색어를 입력하세요: ')
url = f'https://search.naver.com/search.naver?sm=tab_hty.top&where=view&query={quote_plus(search)}'
http = urllib3.PoolManager()
html = http.request('GET', url)
soup = BeautifulSoup(html.data, "lxml")

total = soup.select('.api_txt_lines.total_tit')
searchlist = []

for i in total:
    temp = []
    temp.append(i.text)
    temp.append(i.attrs['href'])
    searchlist.append(temp)

f = open(f'{search}.csv', 'w', encoding = 'utf-8', newline = '') #저장할 때 한줄 추가
csvWriter = csv.writer(f)

for i in searchlist:
    csvWriter.writerow(i)

f.close()

print('완료되었습니다')
