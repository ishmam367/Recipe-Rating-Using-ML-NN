
noOfServings = int(input("Number of servings: "))
noOfIngredients = int(input("Number of ingredients: "))
noOfInstructions = int(input("Number of preparation steps: "))
rating = None

# noOfServings =
# noOfIngredients =
# noOfInstructions =

from classifier import classify

rating = classify(noOfServings,noOfIngredients,noOfInstructions)

print(rating[0])