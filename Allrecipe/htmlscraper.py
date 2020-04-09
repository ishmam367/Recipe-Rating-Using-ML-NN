from bs4 import BeautifulSoup
import datetime
import time
import urllib
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

urllist = set()
with open('recipe_links.txt', 'r') as source:
    for link in source.readlines():
        urllist.add(link)
    source.close()

user_agent = 'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)'


print(str(datetime.datetime.now().time()))

count = 50386
try_again_counter = 0

urllist = list(urllist)
print(len(urllist))

while count<len(urllist):  
    url = urllist[count]  
   
    try:
        request = urllib.request.Request(url,headers={'User-Agent': user_agent})
        html = urlopen(request)

    except HTTPError as e:
        print(e,'trying again ',try_again_counter+1)
        try_again_counter += 1
        if try_again_counter > 2:
            count += 1
            try_again_counter = 0
        continue

    except URLError as e:
        print(e,'trying again ',try_again_counter+1)
        try_again_counter += 1
        if try_again_counter > 2:
            count += 1
            try_again_counter = 0
        continue
    
    except ConnectionResetError as e:
        print(e, 'trying again ', try_again_counter+1)
        try_again_counter += 1
        if try_again_counter > 2:
            count += 1
            try_again_counter = 0
        continue
    
    except:
        print('trying again ', try_again_counter+1)
        try_again_counter += 1
        if try_again_counter > 2:
            count += 1
            try_again_counter = 0
        continue
    

    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        # #Save entire html for backup
        with open('html/'+str(count)+'.html', "w") as htmlFile:
            htmlFile.write(str(bs.body))
        htmlFile.close()

    if count%100==0:
        print(str(count)+' done. Time: '+str(datetime.datetime.now().time()))
    
    count += 1
    try_again_counter = 0
    time.sleep(1)

print("Ended at: ",end='')
print(str(datetime.datetime.now().time()))
