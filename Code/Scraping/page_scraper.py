from bs4 import BeautifulSoup

from urllib.request import urlopen
from urllib.error import HTTPError, URLError


urllist = []
with open('recipe_links.txt', 'r') as source:
    for link in source.readlines():
        urllist.append(link)
    source.close()


count = 0
while count<26347:  
    url = urllist[count]  
    
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    except URLError as e:
        print(e)
    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        try:
            #The title for the recipe
            title = bs.find('h1', {'itemprop':'name'})
            if (title == None) or (len(str(title)) >= 100):
                title = bs.head.find('title').get_text().split('|')[0]
                title = title.replace('&#045', '-')
            else:    
                title = title.get_text()
           
      
            #Just the name(s) of the author(s)
            author = bs.find('a',{'class':'contributor'})
            if author == None:
                author = ''
            else:
                author = author.attrs['title']
       

            #Date (M-Y) the recipe was published
            pub_date = bs.body.find('span', {'class':"pub-date"})
            if pub_date == None:
                pub_date = ''
            else:
                pub_date = pub_date.get_text()
            
    
            #Score of recipe out of 4
            rating = bs.find('span', {'class':'rating'})
            if rating == None:
                rating = ''
            else:
                rating = rating.get_text()
            
        

            #The number of users having reviewed
            reviews = bs.find('span', {'class':'reviews-count', 'itemprop':'reviewCount'})
            if reviews == None:
                reviews = ''
            else:
                reviews = reviews.get_text()

       
           
            #Percentage of Make It Again score
            make_again = bs.find('div',{'class':'prepare-again-rating'})
            if make_again == None:
                make_again = '' 
            elif make_again.span == None:
                make_again = make_again.get_text()
            else:
                make_again = make_again.span.get_text()

        
          
            #Description of the recipe
            description = bs.find('div', {'class':'dek','itemprop':'description'})
            if description == None:
                description = ''
            elif description.p == None:
                description = description.get_text()
            else:
                description = description.p.get_text()


       
            #How many servings of the food
            servings = bs.find('dd',{'class':'yield','itemprop':'recipeYield'})
            if servings == None:
                servings = ''
            else:            
                servings = servings.get_text()


       
            #List of ingredients with their amounts, raw
            ingredients = { }   # {group:[list of ingredients]}
            ingredient_list = bs.find_all('li',{'class':'ingredient-group'})
            for item in ingredient_list:
                if item.strong == None:
                    ingredients['ingredients '+str(ingredient_list.index(item))] = list(item.ul.find_all('li', {'class':'ingredient'}))
                else:
                    ingredient_group_title = item.strong.get_text().strip(':')
                    ingredients[ingredient_group_title] = list(item.ul.find_all('li', {'class':'ingredient'}))




            #Preparation instructions, raw
            preparation = {}
            instructions = bs.find('div', {'class':'instructions', 'itemprop':'recipeInstructions'})
            
            instructions = instructions.find_all('li', {'class':'preparation-group'})
            
            for item in instructions:
                if item.strong == None:
                    preparation['preparation '+ str(instructions.index(item))] = list(item.find_all('li',{'class':'preparation-step'}))
                    
                else:
                    preparation_group_title = item.strong.get_text().strip(':')
                    preparation[preparation_group_title] = list(item.find_all('li',{'class':'preparation-step'}))

            
            # Chef's notes, raw
            chef_note = bs.find('div', class_='chef-notes').get_text()


        except AttributeError as e:
            print (str(count) + " failed. " + str(e))
            count += 1
            continue
        

        #Save to CSV with raw data.
        with open('recipe_till_26346.csv','a') as file:
            file.write('"'+title+'","'+author+'",'+pub_date+','+rating+','+
                reviews+','+make_again+',"'+description+'",'+
                servings+','+str(ingredients)+','+str(preparation)+','+chef_note+'\n')
        file.close()


        #Save entire html for backup
        with open('html/'+str(count)+'.html', "w") as htmlFile:
            htmlFile.write(str(html))
        htmlFile.close()
    
    count += 1   

    print(str(count) + " recipes done.")
