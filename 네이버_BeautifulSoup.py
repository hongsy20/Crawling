import urllib.request
import urllib.parse #한글로 치면 오류나므로 한글을 아스키코드로 바꿔주는 모듈
from bs4 import BeautifulSoup

baseurl = 'https://search.naver.com/search.naver?query=%ED%8C%8C%EC%9D%B4%EC%8D%AC&nso=&where=blog&sm='
plusUrl = input('검색어를 입력하세요: ')
url = baseurl + urllib.parse.quote_plus(plusUrl) #plusUrl에 한글이 들어오면 바꿔줌

html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')

title = soup.find_all(class_='api_txt_lines total_tit')

for i in title:
    print(i.text)
    print(i.attrs['href'])
    print()
    
