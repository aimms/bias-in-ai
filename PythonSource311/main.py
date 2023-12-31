# Set up feedback system

from typing import List
from typing import Tuple

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import json

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# Left as part of global initialization.
def my_init_proc():
    # binds the global data, but for what purpose?
    # According to ChatGPT, binder is a tool for tracking the progress of students.
    # That is why this statement is commented out.
    # binder.bind(globals())

    # Get the same results each time
    np.random.seed(0)

# Actually execute the initialization.
my_init_proc()

# Load the training data
def read_data_csv():
    data = pd.read_csv("./data.csv")
    comments = data["comment_text"]
    target = (data["target"]>0.7).astype(int)
    print(type(comments))
    print(type(target))

    return comments, target

# read_data_csv()

def train_on_data(comments, target):
    # Break into training and test sets
    # train_test_split is part of the scikit-learn (sklearn) package.
    # train_test_split is explained at https://builtin.com/data-science/train-test-split
    comments_train, comments_test, y_train, y_test = train_test_split(comments, target, test_size=0.30, stratify=target)
    # Preview the dataset
    print("Sample toxic comment:", comments_train[22])
    print("Sample not-toxic comment:", comments_train[17])

    # Get vocabulary from training data
    vectorizer = CountVectorizer()
    vectorizer.fit(comments_train)

    # Get word counts for training and test sets
    X_train = vectorizer.transform(comments_train)
    X_test = vectorizer.transform(comments_test)

    # return comments_train, comments_test, y_train, y_test
    return vectorizer, X_train, X_test,  y_train, y_test


# Function to classify any string
def classify_string(string, investigate=False):
    prediction = classifier.predict(vectorizer.transform([string]))[0]
    if prediction == 0:
        print("NOT TOXIC:", string)
    else:
        print("TOXIC:", string)


# Function to classify any string
def classify_this_string(string, classifier, vectorizer, investigate=False):
    prediction = classifier.predict(vectorizer.transform([string]))[0]

    return prediction

# First define the structure of observations.
# Usually an observation has many attributes, but we use only these two here.
class Observation(BaseModel):
    comment: str  # An observed text
    target: int  # A verified zero / one whether this text is considered toxic.

# The input class for the Bias_in_AI app is essentially a row of observations, and a value to predict.
class BiasInAIInputTuples(BaseModel):
    no_observations: int             # not really needed, but used for an internal check.
    observations: list[Observation]  # length is no_observations
    new_comment: str                 # determine for this string, whether it is toxic or not.

# The output class for the Bias_in_AI app:
class BiasInAIOutput(BaseModel):
    is_toxic: int  # 0: not toxic, 1: toxic.


# Create the application object
app = FastAPI()

# Define the actual function to learn and predict which is also the method
@app.post("/compute", response_model=BiasInAIOutput)
async def compute(inp: BiasInAIInputTuples) -> BiasInAIOutput:
    # Copy inp first to a couple of locals.

    obs = inp.observations

    # convert the row of observations to two parallel lists.
    trace_number = 0  # up to trace inputs.
    comments = []
    targets = []
    item_pos = 0
    for item_pos in range(inp.no_observations):
        if item_pos < trace_number:
            print(obs[item_pos].comment + " is toxic: " + str(obs[item_pos].target))
        comments.append(obs[item_pos].comment)
        targets.append(obs[item_pos].target)

    print("comments len = " + str(len(comments)))
    print("targets   len = " + str(len(targets)))

    item_pos = 0
    for item_pos in range(trace_number):
        print(str(targets[item_pos]) + " is toxicity of " + comments[item_pos])

    new_comment = inp.new_comment

    # Get the training and test data as globals.
    vectorizer, X_train, X_test, y_train, y_test = train_on_data(comments, targets)
    print("train on data called")

    # Train a model and evaluate performance on test dataset
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy:", score)
    print("new_comment: "+new_comment)

    # Print the ten most significant toxic words.
    coefficients = pd.DataFrame({"word": sorted(list(vectorizer.vocabulary_.keys())), "coeff": classifier.coef_[0]})
    last_ten = coefficients.sort_values(by=['coeff']).tail(10)
    print(last_ten)

    loc_is_toxic = classify_this_string(new_comment,classifier,vectorizer)
    print("is toxic: ", loc_is_toxic)

    out = BiasInAIOutput(
        is_toxic = loc_is_toxic
    )

    return out


if __name__ == "__main__":
    uvicorn.run("main:app", host="", port=8000, log_level="info")

