# NLP Data Science Technical Interview
![NLP Data Science](NLP_data_science.jpg)


## Problem
A fictional B2B company sells products to online shops and wants to create a list of potential customers.
They scrapped a large number of major German web sites and labelled some of the data with a flag denoting if the web site is an online shop or not.


### Task 1
Develop a classifier which is able to predict whether a web site is an online shop by looking at the HTML content of its main page.

### Task 2
Using this classifer, create predictions for each of the web sites which are unclassified so far ("dataset 2"). Provide the prediction as a CSV file containing the domain name and a flag that denotes if the respective web site is an online shop.

### Task 3
Explain your approach and its technical details to our team.

### Task 4
Alas, the VP Sales of the company does not trust black box models and thus wants to understand what the model learned and how it comes to its decisions. In order to convince him, illustrate which content a web page needs to contain or how it needs to be structured to be classified as an online shop by your model. The VP Sales is a non-technical person and does not have deep knowledge about data analysis, so keep this part of the presentation as easy to understand as possible.

**NOTE**: We will combine task 3 into task 1, so that as we build the solution, it will also be explained.


## Principles
To solve this problem we will stick to a few basic principles.

* The principle of finding the **lowest hanging fruit**. That is, we will find the the thing that brings the most value with the least effort.
* One the core principles of **Agile**. We will solve the problem **incrementally** and **iteratively**. We don't want to get bogged down in perfecting one aspect and using up all of our time obsessing over one thing.

## Installation
It is recommened to use a virtual environment to install the required packages. Once you create your virtual environment and activated it, you can install the required packages by running the following command:

```pip install -r requirements.txt```

Then you can use the `notebook.ipynb` file to run the interview.