🚀 Winning Space Race with Data Science

This project looks at SpaceX launch data to understand what makes a first-stage landing successful and whether we can use historical launch information to predict the outcome of a future landing.

The main goal was to collect real launch data, clean it, explore it from different angles, build visualizations, and finally train machine learning classification models to predict the landing outcome.

🎯 Project Goal

SpaceX can reduce launch costs by reusing the first stage of its Falcon 9 rockets. The presentation highlights a launch cost of about $62 million, compared with other providers that can sometimes cost more than $165 million.

That made the main questions of this project:

Can we predict whether the first stage of a new launch will land successfully?

What factors are related to a successful landing?

Which launch site, orbit type, payload, or booster information can help us understand launch success?

Which machine learning model performs best for predicting the landing outcome?

🔄 Project Flow

flowchart TD
    A[Collect SpaceX Data] --> B[Clean & Prepare Data]
    B --> C[Create Landing Class]
    C --> D[Exploratory Data Analysis]
    D --> E[SQL Analysis]
    E --> F[Interactive Visualizations]
    F --> G[Prepare ML Dataset]
    G --> H[Train Classification Models]
    H --> I[GridSearchCV]
    I --> J[Evaluate Models]
    J --> K[Select Best Model]
    K --> L[Interpret Results & Limitations]

📥 1. Data Collection

The project combines data from two main sources.

SpaceX REST API

Launch information was collected through the SpaceX REST API and converted from JSON into a Pandas DataFrame.

SpaceX REST API
      ↓
    JSON
      ↓
Pandas DataFrame

Wikipedia Web Scraping

Additional Falcon 9 launch information was collected from the Wikipedia Falcon 9 launch page.

Wikipedia
    ↓
   HTML
    ↓
BeautifulSoup
    ↓
Pandas DataFrame

The two approaches allowed the project to bring together the information needed for the analysis.

🧹 2. Data Wrangling

The raw landing outcomes were converted into a simpler classification target called Landing Class.

The basic idea was:

Successful landing outcome
          ↓
      Landing Class
          ↓
           1

Unsuccessful landing outcome
          ↓
      Landing Class
          ↓
           0

For example, successful outcomes such as ground-pad, drone-ship, and ocean landings were treated as successful landing classes, while unsuccessful outcomes were treated as failures.

This made the data easier to use for classification models.

🔎 3. Exploratory Data Analysis

Before building the machine learning models, the data was explored using visualizations and SQL.

Visual Analysis

Several relationships were investigated:

Flight Number vs. Launch Site

Payload vs. Launch Site

Success Rate vs. Orbit Type

Flight Number vs. Orbit Type

Payload vs. Orbit Type

Yearly launch success trend

EDA Flow

flowchart LR
    A[Clean Dataset] --> B[Explore Variables]
    B --> C[Create Charts]
    C --> D[Find Patterns]
    D --> E[Identify Useful Features]
    E --> F[Prepare for Prediction]

📊 4. What the EDA Showed

Launch Sites

The analysis identified four launch sites:

CCAFS LC-40

VAFB SLC-4E

KSC LC-39A

CCAFS SLC-40

The presentation shows that three sites are on the eastern side and one is on the western side, with all four located in the south.

Launch Success Trend

The yearly trend showed that launch success increased considerably from around 2013 through 2020, although the trend was not perfectly consistent every year.

Orbit Types

The analysis showed particularly high success rates for several orbit types, including ES-L1, GEO, HEO, and SSO. The presentation also notes that SO orbit had a zero success rate in the analyzed data.

Payload

The analysis found that payload and landing success were related differently depending on the launch site and orbit.

The dashboard analysis also highlighted the 3000–4000 kg payload range as having the largest success rate in the presented results.

🗃️ 5. SQL Analysis

SQL was used to answer specific questions from the launch database.

Examples included:

What are the unique launch sites?

Which booster versions carried the maximum payload?

How many missions were successful or failed?

Which boosters successfully landed on a drone ship
with a payload between 4000 and 6000 kg?

How do landing outcomes rank between selected dates?

Some results from the analysis included:

100 successful mission outcomes

1 failure mission outcome

First successful ground-pad landing: 01-05-2017

Average payload for F9 v1.1: approximately 2534.67 kg

The project also identified four booster versions associated with successful drone-ship landings for payloads between 4000 and 6000 kg:

F9 FT B1022

F9 FT B1026

F9 FT B1021.2

F9 FT B1031.2

🗺️ 6. Interactive Map with Folium

Folium was used to make the launch-site analysis easier to understand visually.

The map included:

Launch-site markers

Circles around launch locations

Clusters for successful and failed launches

Lines showing distances and nearby locations

flowchart TD
    A[Launch Site Coordinates] --> B[Folium Map]
    B --> C[Launch Site Markers]
    B --> D[Success/Failure Clusters]
    B --> E[Distance & Proximity Analysis]

The presentation specifically examined KSC LC-39A and reported approximate distances of:

4.16 km to the nearest shuttle landing facility

Less than 1 km to the nearest highway

Around 6.5 km to the coastline

📈 7. Plotly Dash Dashboard

A Plotly Dash dashboard was created to make the analysis interactive.

The dashboard allows users to explore:

Launch success distribution by launch site

Payload and success relationships

Different launch sites using a dropdown

Payload ranges using a range slider

Visual relationships using charts

Dashboard Flow

User selects Launch Site
          ↓
Dashboard filters data
          ↓
Pie Chart shows success distribution
          ↓
Payload range can be adjusted
          ↓
Scatter Plot shows payload vs. success

One of the main dashboard findings was that KSC LC-39A had the highest launch success ratio at 76.9% in the presented analysis.

🤖 8. Predictive Analysis

After the exploratory analysis, the project moved to classification.

The machine learning workflow was:

flowchart LR
    A[Prepare Data] --> B[Create Target Class]
    B --> C[Standardize Data]
    C --> D[Train/Test Split]
    D --> E[Define Models]
    E --> F[GridSearchCV]
    F --> G[Find Best Parameters]
    G --> H[Evaluate Models]

The dataset contained:

90 rows

83 columns

With an 80/20 split:

72 training records

18 testing records

🔧 9. Model Selection with GridSearchCV

Several classification models were trained and tuned using GridSearchCV.

The purpose was to search through different model parameters and find a good-performing configuration.

The presentation identifies the Decision Tree as the preferred model based on the classification results.

The reported Decision Tree results were:

Training Accuracy: 0.90
Testing Accuracy: 0.94

The selected parameters included:

criterion = gini
max_depth = 8
max_features = auto
min_samples_leaf = 2
min_samples_split = 10
splitter = random

🌳 10. Decision Tree Results

The Decision Tree was able to distinguish between the different classes in the test results.

However, the confusion matrix also showed an important issue:

False positives were the major problem.

This matters because an incorrect prediction could affect the estimation of whether a future rocket launch is likely to land successfully, which could in turn affect decisions around launch bidding and cost estimates.

⚠️ 11. Limitations

The most important limitation of this project is the size and shape of the dataset.

90 observations
       +
83 features
       ↓
More features than samples
       ↓
Higher risk of overfitting

With only 72 training records and 18 testing records, there is not enough data to confidently generalize the model to every future launch.

This means the reported 94% testing accuracy should be interpreted carefully rather than treated as proof that the model will perform equally well on new real-world launches.

🛠️ 12. Possible Improvements

The presentation suggests several ways the model could be improved:

Get More Data

More launch records would give the models more examples to learn from.

Remove Less Useful Features

EDA can help identify highly correlated or less important variables and reduce unnecessary columns.

Regularization

Regularization could help control overfitting.

Dimension Reduction

PCA (Principal Component Analysis) could be tested to reduce the number of dimensions.

flowchart TD
    A[Small Dataset + Many Features] --> B{Possible Solutions}
    B --> C[Collect More Data]
    B --> D[Remove Unimportant Features]
    B --> E[Use Regularization]
    B --> F[Try PCA]
    C --> G[Reduce Overfitting Risk]
    D --> G
    E --> G
    F --> G

💡 Key Takeaways

This project was not just about building a machine learning model. It followed the complete data science process:

Real-World Question
        ↓
Data Collection
        ↓
Data Wrangling
        ↓
EDA
        ↓
SQL Analysis
        ↓
Visualization
        ↓
Interactive Dashboard
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Business Interpretation

The analysis suggests that launch site, payload, orbit type, booster information, and launch history can provide useful information for understanding landing outcomes.

At the same time, the small dataset means the model needs to be treated carefully and improved with more data before relying on it for important real-world decisions.

🧰 Tools & Technologies

Python

Pandas

NumPy

Requests

BeautifulSoup

SQL

SQLite

Matplotlib

Seaborn

Folium

Plotly

Plotly Dash

Scikit-learn

GridSearchCV

Jupyter Notebook

📁 Suggested Project Structure

winning-space-race-with-data-science/
│
├── data/
│   └── spacex_data.csv
│
├── notebooks/
│   └── SpaceX_Launch_Analysis.ipynb
│
├── dashboard/
│   └── app.py
│
├── maps/
│   └── launch_sites_map.html
│
├── images/
│   ├── eda/
│   ├── dashboard/
│   └── model_results/
│
├── README.md
└── requirements.txt

👤 Author

Abdul Raheem Bin Riyaz Khan

IBM Data Science Capstone Project

📌 Project Status

Completed as a data science capstone project.

The project demonstrates the complete workflow from collecting SpaceX launch data to exploratory analysis, SQL investigation, interactive visualization, and machine learning classification.

The main area for future improvement is increasing the amount of training data and reducing the risk of overfitting before using the model for real-world prediction.

