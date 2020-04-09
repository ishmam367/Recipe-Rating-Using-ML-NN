from bs4 import BeautifulSoup
import datetime
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

urllist = []
with open('recipe_links.txt', 'r') as source:
    for link in source.readlines():
        urllist.append(link)
    source.close()

print(str(datetime.datetime.now().time()))

count = 13125
while count<len(urllist):  
    url = urllist[count]  
   
    try:
        html = urlopen(url)

    except HTTPError as e:
        print(e)

    except URLError as e:
        print(e)

    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        # #Save entire html for backup
        with open('html/'+str(count)+'.html', "w") as htmlFile:
            htmlFile.write(str(bs.body))
        htmlFile.close()
    
    count += 1
    
    if count%1000==0:
        print(str(count)+' done. Time: '+str(datetime.datetime.now().time()))


print("Ended at: ")
print(str(datetime.datetime.now().time()))
