from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import time
url = 'http://movie.naver.com/movie/sdb/rank/rmovie.nhn'
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")
result = soup.find_all("div",attrs={"class":"tit3"})

urlList = []
# for i in range(len(result)):
for i in range(0,5):
    aList = result[i].find_all('a')
    for aTag in aList:
        urlList.append(aTag.get('href'))

        
url1 = "https://movie.naver.com/movie/bi/mi/photoViewPopup.nhn?movieCode="
last = []
for i in range(len(urlList)):
    temp = urlList[i].split('code=')
    url2 = url1 + temp[1]
    page = urlopen(url2)
    soup = BeautifulSoup(page, "html.parser")
    kresult = soup.find_all("img",attrs={"id":"targetImage"})
    for aTag in kresult:
        last.append(aTag.get('src'))
print(last)

for i in range(len(last)):
    urlretrieve(last[i], str(i)+".jpg")
    time.sleep(2)