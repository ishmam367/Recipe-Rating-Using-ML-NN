import pandas as pd
import numpy as np


dataset=pd.read_csv('../Allrecipe/recipe_v2.csv')
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()

#title,preptime,cooktime,totaltime,servings,ingredients,steps,calories,rating,RatingClass

# dataset = dataset.astype({"title":'object',"preptime":'int64',"cooktime":'int64',"totaltime":'int64',"servings":'float64',"ingredients":'int64',"steps":'int64',"calories":'float64',"rating":'float64', "RatingClass":'category'})
dataset = dataset[['title','preptime','cooktime','totaltime','servings','ingredients','steps','calories','rating']]

length = len(dataset['rating'])

dataset = dataset.assign(RatingClass=pd.Series(np.zeros(length)).values)
# Reviews to ClassofRating
for i, row in dataset.iterrows():
    ratingValue = row.loc['rating']

    if type(ratingValue) == str:
        ratingValue = float(ratingValue)
    
    if (ratingValue>=0) and (ratingValue<3.5):
        dataset.loc[i,'RatingClass'] = "Average"     #Denotes average
    elif (ratingValue>=3.5) and (ratingValue<4.5):
        dataset.loc[i,'RatingClass'] = "Good"     #Denotes good
    elif (ratingValue>=4.5) and (ratingValue<=5):
        dataset.loc[i,'RatingClass'] = "Excellent"     #Denotes excellent


for i, item in dataset.iterrows():
    preptime = item.loc['preptime']
    cooktime = item.loc['cooktime']
    totaltime = item.loc['totaltime']


    # print(preptime,cooktime,totaltime)

    prep = preptime.split()
    preptime = 0
    for x,p in enumerate(prep):
        if p.isalpha():
            continue
        else:
            if 'm' in prep[x+1]:
                preptime += int(p)
            elif 'h' in prep[x+1]:
                preptime += int(p) * 60
    
    cook = cooktime.split()
    cooktime = 0
    for x,p in enumerate(cook):
        if p.isalpha():
            continue
        else:
            if 'm' in cook[x+1]:
                cooktime += int(p)
            elif 'h' in cook[x+1]:
                cooktime += int(p) * 60
    

    total = totaltime.split()
    totaltime = 0
    for x,p in enumerate(total):
        if p.isalpha():
            continue
        else:
            if 'm' in total[x+1]:
                totaltime += int(p)
            elif 'h' in total[x+1]:
                totaltime += int(p) * 60
    
    # print(preptime,cooktime,totaltime)

    dataset.loc[i,'preptime'] = preptime
    dataset.loc[i,'cooktime'] = cooktime
    dataset.loc[i,'totaltime'] = totaltime
    

dataset.to_csv('../Allrecipe/data2.csv')
