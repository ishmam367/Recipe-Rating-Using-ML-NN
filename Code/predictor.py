import pandas as pd
import numpy as np
import string
from sklearn.naive_bayes import MultinomialNB
from tokenizer import spacy_tokenizer
from collections import Counter
from sklearn.model_selection import ShuffleSplit

data = pd.read_csv('../recipe.csv', encoding='utf-8', delimiter="&&",error_bad_lines=False, engine='python')



data = data[['reviews', 'make_again', 'servings', 'ingredients', 'preparation', 'rating']]

newDataset = pd.DataFrame(columns=['Reviews', '% Make Again', 'Servings', 'Ingredients', 'Instructions', 'Rating'])

for i, item in data.iterrows():
        prep = item.loc['preparation']
        ingredient = item.loc['ingredients']
        
        if type(ingredient) != str:
                ingredient = str(ingredient)
        if type(prep) != str:
                prep = str(prep)
                
        puncuations = string.punctuation
        puncuations += '1234567890'
        puncuations.replace('.','')

        for stuff in puncuations:              
                ingredient = ingredient.strip(stuff)
                prep = prep.strip(stuff)
        
        ingredient = ingredient.strip(".")
        prep = prep.replace('\\n',' ')
        
        removals = ['tsp.','tbsp.', 'Tsp.', 'Tbsp.']
        for items in removals:
                prep = prep.replace(items, ' ')

        ingredient = ingredient.split('\\n')
        prep = prep.split('. ')

        # print(ingredient,'\n',prep)
        
        noOfIngredients = len(ingredient) - 1
        noOfInstructions = len(prep)

        # print("ingredient "+str(noOfIngredients),"prep "+str(noOfInstructions))

        newDataset = newDataset.append({'Reviews': item.loc["reviews"], '% Make Again': item.loc["make_again"], 'Servings': item.loc["servings"], 'Ingredients': noOfIngredients, 'Instructions':noOfInstructions, 'Rating':item.loc['rating']}, ignore_index=True)
        print(i)

newDataset.to_csv("../data.csv")


# split = ShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

# for train_index, test_index in split.split(data):
#         trainSet = data.loc[train_index]
#         testSet = data.loc[test_index]

# trainSet.to_csv('../train.csv',sep='|',encoding='utf-8', index=False)
# testSet.to_csv('../test.csv',sep='|',encoding='utf-8', index=False)

# ingredients = []
# preparations = []

# for i,item in data.iterrows():
#     prep = item['preparation']
#     ingredient = item.loc['ingredients']
    
#     if type(prep) == str:
#         prep = prep.strip('{').strip('}')
#     else:
#         prep = str(prep).strip('{').strip('}')

#     if type(ingredient) == str:
#         ingredient = ingredient.strip('{').strip('}')
#     else:
#         ingredient = str(ingredient).strip('{').strip('}')

#     ingredients.extend(spacy_tokenizer(ingredient))
#     preparations.extend(spacy_tokenizer(prep))


#     count += 1
#     if count % 10 == 0:
#         print(str(count)+' done.')

# prep_vocabulary = Counter(preparations).most_common(10000)
# ingred_vocabulary = Counter(ingredients)

