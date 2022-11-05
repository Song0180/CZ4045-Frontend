# CZ4045-Frontend

This repository stores the web app [LOL Sentiment Analyzer](https://song0180-cz4045-frontend-data-app-e1utyw.streamlit.app/) implemented using [Streamlit](https://streamlit.io/), which is a powerful tool to create data apps.

## Installation

### Demo

The web app is publicly available. If you just want to try the demo, click [here](https://song0180-cz4045-frontend-data-app-e1utyw.streamlit.app/).

### Run locally

1. Follow [this official guide](https://docs.streamlit.io/library/get-started/installation) to install Streamlit and essential dependencies.
2. Install required dependencies

```shell
  cd CZ4045-Frontend
  pip install requirements.txt
```

3. Run the app locally

```shell
streamlit run data_app.py
```

## Files

- `data_app.py`: the python script that runs the app.
- `requirements.txt`: the list of python dependencies.
- `model_1_rf.sav`: random forest model for prediction of neutral/non-neutral sentiment.
- `model_2_svm.sav`: support vector machine model for prediction of positive/negative sentiment.
- `vectorizer.pickle`: TF-IDF Vectorizer.
- `vectorizer2.pickle`: TF-IDF Vectorizer.
- `main_data.csv`: the dataset used for training the models.
- `sample_data.csv`: the evaluation dataset for testing the app. It only contains LOL related reddit post titles.

## How to use the web app

### Previous Sentiments

This section displays a table of data in the `main_data.csv`, which contains a list of LOL related reddit post titles with corresponding post date and manually labelled sentiment value.

### Previous Trends

This section displays a stacked column chart based on the data in the `main_data.csv`. It shows sentiment polarity on a monthly basis in posts with non-neutral sentiment values.

### Add new data

This section allows the user to add custom data of LOL related reddit post titles.

1. To add new data, the user **must** specify the date of the new posts.
1. The new data should be stored in a `.csv` file, which should only contain one column of post titles
1. The `.csv` file should be uploaded via the file uploader either by drag and drop or clicking the `Browse File` button.

Once a valid `.csv` file is uploaded, the app will attach the date specified to each data item (i.e. a post title) and predict the corresponding sentiment value (-1, 0, or 1).

After the calculation is done, the user will be able to see the following sections:

#### Sentiment Prediction for New Data

This section displays a pie chart that shows the percentage of each sentiment value in the new data.

#### Identified Text with Sentiments

This section displays a table of LOL related reddit post titles with the user-specified date and predicted sentiment value.

#### New Sentiment Trends

This section displays a stacked column chart of the overall sentiment trends, which is similar to what the user can see previously in the Previous Trends section. The user will notice that the polarity of the new data over the month of the user-specified date is appended at the right most of the chart, which reflects the latest polarity trends.

**NOTE that While you are free to add any data that contains LOL related reddit post titles, we have included a sample data set `sample_data.csv` in this repository and we highly recommend using it for testing purpose.**
