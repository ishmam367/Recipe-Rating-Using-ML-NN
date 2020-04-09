from bs4 import BeautifulSoup

count = 0
fails = 0

while count<51434:  
    
    try:              # SET PATH TO HTML FOLDER
        html = open('html/'+str(count)+'.html', 'r',encoding='utf-8')
        # html = open('html/10.html','r')
    
    except FileNotFoundError as e:
        print(e)
        count += 1
        continue
    
    else:    
        bs = BeautifulSoup(html.read(), 'html.parser')
        try:
            #The title for the recipe
            title = bs.find('h1', {'class':'headline heading-content'})
            if (title == None) or (len(str(title)) >= 100):
                title = bs.find('h1', {'itemprop':'name'})  
                if title == None:
                    title = '' 
                else: 
                    title = title.get_text()
            else:    
                title = title.get_text()
           
       
            #Score of recipe out of 5
            rating = bs.find('span', {'class':'review-star-text'})
            if rating == None:
                rating = bs.find('span',{'class':'aggregate-rating'})
                if rating != None:
                    rating = rating.meta.attrs['content']#.get_text()
            else:
                rating = rating.get_text().replace('Rating:','').replace('stars', '').strip()
           
            if rating == 'Unrated' or rating == None:
                rating = 0
        
            #Preparation time and servings
            timeandservings = bs.find_all('div',{'class':'recipe-meta-item'})
            preptime = ''
            cooktime = ''
            totaltime = ''
            servings = ''
            
            if (timeandservings == None) or (len(timeandservings)==0):
                timeandservings = bs.find_all('li',{'class':'prepTime__item'})
                timeandservings = [item.find('p',{'class':'prepTime__item--type'}) for item in timeandservings]#timeandservings.find_all('p',{'class':'prepTime__item--type'})
                if len(timeandservings)>0:
                    timeandservings.pop(0)
                    for p in timeandservings:
                        if p.get_text().strip() == "Prep":
                            preptime = p.find_next_sibling().span.get_text().strip()
                        elif p.get_text().strip() == "Cook":
                            cooktime = p.find_next_sibling().span.get_text().strip()  
                        elif p.get_text().strip() == "Ready In":
                            totaltime = p.find_next_sibling().span.get_text().strip()  

                servings = bs.find('meta',{'id':'metaRecipeServings'})
                if servings==None:
                    servings = ''
                else:
                    servings = servings.attrs['content']
            else: 
                for div in timeandservings:
                    # print(div)
                    if div.find('div',{'class':'recipe-meta-item-header'}).get_text().strip().strip("\"").strip() == "prep:":
                        preptime = div.find('div',{'class':'recipe-meta-item-body'}).get_text().strip()
                    elif div.find('div',{'class':'recipe-meta-item-header'}).get_text().strip().strip("\"").strip() == 'cook:':
                        cooktime = div.find('div',{'class':'recipe-meta-item-body'}).get_text().strip()   
                    elif div.find('div',{'class':'recipe-meta-item-header'}).get_text().strip().strip("\"").strip() == 'total:':
                        totaltime = div.find('div',{'class':'recipe-meta-item-body'}).get_text().strip()
                    elif div.find('div',{'class':'recipe-meta-item-header'}).get_text().strip().strip("\"").strip() == 'Servings:':
                        servings = div.find('div',{'class':'recipe-meta-item-body'}).get_text().strip()



            #List of ingredients with their amounts
            temp_list = [ ]

            ingredient_list = bs.find_all('li',{'class':'ingredients-item'})
            if (ingredient_list == None) or (len(ingredient_list)==0):
                ingredient_list = bs.find_all('span',{'class':'recipe-ingred_txt added'})
                for item in ingredient_list:
                    temp_list.append(item.get_text()) #-1 for this type maybe
            else:
                for item in ingredient_list:
                    temp_list.append(item.find('span', {'class':'ingredients-item-name'}).get_text())

            ingredient_count = len(temp_list)
            
            #Number of Steps
            instructions = bs.find_all('li',{'class':'subcontainer instructions-section-item'})
            if instructions == None or len(instructions) == 0:
                instructions = bs.find_all('span',{'class':'recipe-directions__list--item'})
                # if instructions == None:
                #     instructions = ''
            
            steps = len(instructions)

            #per serving calories
            nutrition = bs.find('div',{'class':'partial recipe-nutrition-section'})
            if nutrition == None:
                calories = bs.find('span',{'itemprop':'calories'}).get_text().split()[0]
                if calories == None:
                    calories = ''
            else:
                nutrition = nutrition.find('div',{'class':'section-body'}).get_text()
                calories = nutrition.split(';')[0].split()[0]


            
        except AttributeError as e:
            fails += 1
            print (str(count) + " failed. " + str(e) + '(' + str(fails) + ')')
            count += 1
            continue


        #Save to CSV with raw data.
        with open('recipe_v2.csv','a', encoding='utf-8') as file:
            if float(rating) != 0:
                file.write('\"'+title+'\"'+','+preptime+','+cooktime+','+totaltime+','+servings+','+str(ingredient_count)+','+str(steps)+','+calories+','+rating+'\n')
        
        # print ('Title: {}\t Rating: {}\t PrepTime: {}\nCookTime: {}\t TotalTime: {}\tServings: {}\nIngredients: {}\tInstructions: {}\tCalories: {}'.format(title, rating, preptime, cooktime, totaltime, servings, ingredient_count, steps, calories))
                    

        count += 1   
        print(str(count) + " recipes done. ("+str(fails)+")")

print('failed '+str(fails)+' recipes.')
