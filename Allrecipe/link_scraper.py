from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib
import requests
from itertools import cycle
import traceback
from urllib.error import HTTPError, URLError
import time

url_list = ['https://www.allrecipes.com/recipes/76/appetizers-and-snacks/',
    'https://www.allrecipes.com/recipes/156/bread/',
    'https://www.allrecipes.com/recipes/78/breakfast-and-brunch/',
    'https://www.allrecipes.com/recipes/249/main-dish/casseroles/',
    'https://www.allrecipes.com/recipes/79/desserts/',
    'https://www.allrecipes.com/recipes/276/desserts/cakes/',
    'https://www.allrecipes.com/recipes/367/desserts/pies/',
    'https://www.allrecipes.com/recipes/362/desserts/cookies/',
    'https://www.allrecipes.com/recipes/17562/dinner/',
    'https://www.allrecipes.com/recipes/77/drinks/',
    'https://www.allrecipes.com/recipes/138/drinks/smoothies/',
    'https://www.allrecipes.com/recipes/17561/lunch/',
    'https://www.allrecipes.com/recipes/95/pasta-and-noodles/',
    'https://www.allrecipes.com/recipes/96/salad/',
    'https://www.allrecipes.com/recipes/215/salad/pasta-salad/',
    'https://www.allrecipes.com/recipes/17031/side-dish/sauces-and-condiments/',
    'https://www.allrecipes.com/recipes/81/side-dish/',
    'https://www.allrecipes.com/recipes/94/soups-stews-and-chili/'
    ]



user_agent = 'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)'
i=14
notfoundcount = 0
count = 0
while True:
    url = url_list[i]
    try:
        request = urllib.request.Request(url+'?page='+str(count), headers={'User-Agent': user_agent})
        # print(url+'?page='+str(count))
        html = urlopen(request)
        
        # html = requests.get(url+'?page='+str(count), proxies={'http':proxy, 'https':proxy})
    except HTTPError as e:
        if(notfoundcount>5):
            i += 1
            count = 0
            continue
        notfoundcount += 1
        print(e)
    except URLError as e:
        print(e)
    except:
        print("had connection issue, retrying")
        continue
    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')

        recipes = bs.find_all('article',{'class':'fixed-recipe-card'})

        recipes = [item.h3.a.attrs['href'] for item in recipes]

        recipes = [item + '\n' for item in recipes]
   
   
        with open('recipe_links.txt', "a") as file:
            
            file.writelines(recipes)
    
    
    print (str(count)+' pages done.\n')
    count += 1
    
    time.sleep(1)
    if count == 801:
        i += 1
        count = 0
    if i == 18:
        break
