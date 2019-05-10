# Exploring Airbnb open data following CRISP-DM methodology

## Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

<a name="motivation"></a>
## Project Motivation
This project was created as part of Udacity's Data Scientist for Enterprise nanodegree. Here I have analyzed [Seattle Airbnb Open Data](https://www.kaggle.com/Airbnb/seattle/data) following CRISP-DM methodology. Airbnb data for other cities have the same format. So the same understandings and code can be applied to Airbnb dataset of any other city.

The three business questions which I have tried to answer in this project are as follows:
- What is the seasonal price trend of Airbnb listings in Seattle? When are the most expensive and cheapest times to visit Seattle?
- How does price of Airbnb listings vary in different neigbourhoods?
- What are the most important factors influencing the price of Airbnb listings?

<a name="installation"></a>
## Installation

The code should run using any Python versions 3.*. I used python 3.6.

Libraries Used : numpy, pandas, matplotlib, seaborn, sklearn

If you don't have python already installed, I would suggest you to install the Anaconda distribution of Python, as you will get all the necessary libraries together.

<a name="files"></a>
## File Descriptions
The analysis is divided into 4 files. The name of the files are self-explanatory. Each of the notebooks contains code and explains the detailed analysis performed to arrive at the below mentioned [results](#results) for each of the questions showcased by the notebook titles.
- **Business and Data Understanding.ipynb**
- **Question 1 - Seasonal price trend.ipynb**
- **Question 2 - Price trend by neighborhood.ipynb**
- **Question 3 - Factors influencing price.ipynb**

<a name="results"></a>
## Results
- Prices of Airbnb listings in Seattle are highest from July to September. The cheapest time is at the start of the year from January to March.
- The most expensive neighbourhoods in Seattle are Downtown, Magnolia, Queen Anne, Cascade and West Seattle. 
  Capitol Hill and Downtown neighbourhoods have highest number of listings.
- The most important features which influence prices of Airbnb listings in Seattle are bedrooms, accommodates, bathrooms, beds - all indicating the size of the listing. Room type(Entire Apartment/House, Private Room or Shared Room) and reviews per month are also important features. Location also plays an important role.
  The most important amenities influencing price are Family/Kid Friendly, TV, Indoor Fireplace, Elevator in Building, Hot Tub, Gym and Kitchen.

For a more detailed non-technical discussion, check out [my blog post](https://medium.com/@rehan.ahmar/analyzing-airbnb-open-data-following-crisp-dm-methodology-afbc7b1a9b64).

<a name="acknowledgements"></a>
## Acknowledgements
Thanks to Airbnb and Kaggle for the data, and Udacity for course meterial.
