from bs4 import BeautifulSoup

count = 0
fails = 0

while count<36252:  
    
    try:              # SET PATH TO HTML FOLDER
        html = open('/media/michael/Mikhail/html/'+str(count)+'.html', 'r',encoding='utf-8')
    
    except FileNotFoundError as e:
        print(e)
        count += 1
        continue
    
    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        try:
            #The title for the recipe
            title = bs.find('h1', {'itemprop':'name'})
            if (title == None) or (len(str(title)) >= 100):
                count+=1
                continue
            else:    
                title = title.get_text()
           
      
            #Just the name(s) of the author(s)
            author = bs.find('a',{'class':'contributor'})
            if author == None:
                author = ''
            else:
                author = author.attrs['title']
       

            #Date (M-Y) the recipe was published
            pub_date = bs.find('span', {'class':"pub-date"})
            if pub_date == None:
                pub_date = ''
            else:
                pub_date = pub_date.get_text()
            
    
            #Score of recipe out of 4
            rating = bs.find('span', {'class':'rating'})
            if rating == None:
                rating = ''
            else:
                rating = rating.get_text().split('/')[0]
            
        

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
                make_again = make_again.get_text().strip('%')
            else:
                make_again = make_again.span.get_text().strip('%')

        
          
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


       
            #List of ingredients with their amounts, structure of listed text amounts not consistent
            ingredients = { }   # {group:list of ingredients}
            temp_list = [ ]
            temp_string = ""

            ingredient_list = bs.find_all('li',{'class':'ingredient-group'})

            for item in ingredient_list:

                temp_list.extend(item.ul.find_all('li', {'class':'ingredient'}))
                
                for i,element in enumerate(temp_list):
                    temp_string += element.get_text() + '\n'  #subject to change
                      

                if item.strong == None:
                    ingredients['ingredients '+str(ingredient_list.index(item))] = temp_string
                else:
                    ingredient_group_title = item.strong.get_text().strip(':')
                    ingredients[ingredient_group_title] = temp_string




            #Preparation instructions
            preparation = { }
            temp_list = [ ]
            temp_string = ""

            instructions = bs.find('div', {'class':'instructions', 'itemprop':'recipeInstructions'})
            instructions = instructions.find_all('li', {'class':'preparation-group'})
            
            for item in instructions:
                temp_list.extend(item.find_all('li',{'class':'preparation-step'}))

                for i,element in enumerate(temp_list):
                    temp_string += element.get_text() + '\n'  #subject to change

                if item.strong == None:
                    preparation['preparation '+ str(instructions.index(item))] = temp_string.strip()
                    
                else:
                    preparation_group_title = item.strong.get_text().strip(':')
                    preparation[preparation_group_title] = temp_string.strip()

            
            
        except AttributeError as e:
            fails += 1
            print (str(count) + " failed. " + str(e) + '(' + fails + ')')
            count += 1
            continue
        

        #Save to CSV with raw data.
        with open('../../recipe.csv','a', encoding='utf-8') as file:
            if rating != '0':
                file.write(title+'||'+author+'||'+pub_date+'||'+rating+'||'+
                    reviews+'||'+make_again+'||'+description+'||'+
                    servings+'||'+str(ingredients)+'||'+str(preparation)+'\n')
        


    count += 1   

    print(str(count) + " recipes done. ("+str(fails)+")")

print('failed '+str(fails)+' recipes.')
