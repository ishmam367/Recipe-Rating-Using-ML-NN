from bs4 import BeautifulSoup

from urllib.request import urlopen
from urllib.error import HTTPError, URLError

url = 'https://www.epicurious.com/search/?content=recipe&page='
count = 1
while count<2006:
    try:
        html = urlopen(url+str(count))
    except HTTPError as e:
        print(e)
    except URLError as e:
        print(e)
    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        recipes = bs.find_all('h4',{'class':'hed'})
        recipes = [item.a.attrs['href'] for item in recipes]
        recipes = [ 'https://www.epicurious.com' + 
                        item + '\n' for item in recipes
        ]

        with open('recipe_links.txt', "a") as file:
            
            file.writelines(recipes)
        
    count += 1
    if (count%20 == 0):
        print ('20 pages done.\n')