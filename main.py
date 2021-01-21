import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

siteHeader = st.beta_container()
dataExploration = st.beta_container()
newFeatures = st.beta_container()
modelTraining = st.beta_container()

st.markdown(
      """
      <style>
       .main { background-color: #F5F5F5;}
      </style>
      """,
      unsafe_allow_html=True
  )

@st.cache
def get_data(filename):
    hr_df = pd.read_csv(filename)
    return hr_df

with siteHeader:
    st.title('Welcome to the Awesome project!')
    st.text('In this project I look into ... And I try ... I worked with the dataset from ...')

with dataExploration:
    st.header('Human Resource Dataset')
    st.text('I found this dataset at... I decided to work with it because ...')

    hr_df = get_data('HR_dataset.csv')
    st.write(hr_df)

    st.subheader('Number of projects distribution')
    distribution_numprojects = pd.DataFrame(hr_df['number_project'].value_counts())
    st.bar_chart(distribution_numprojects)

with newFeatures:
    st.header('New features I came up with')
    st.markdown('* **first feature:** this is the explanation')
    st.markdown('* **second feature:** another explanation')
    st.text('Let\'s take a look into the features I generated.')

with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the hyperparameters!')

    selection_col, display_col = st.beta_columns(2)

    max_depth = selection_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value = 20, step = 10)
    number_of_trees = selection_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limit'], index=0)

    selection_col.text('Here is a list of features:')
    selection_col.write(hr_df.columns.values)
    input_feature = selection_col.text_input('Which feature would you like to input to the model?', 'Salary')

    if number_of_trees == 'No limit':
        regr = RandomForestRegressor(max_depth = max_depth)
    else:
        regr = RandomForestRegressor(max_depth = max_depth, n_estimators = number_of_trees)

    X = hr_df[[input_feature]]
    y = hr_df[['left']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    display_col.subheader('Mean absolute error of the model is:')
    display_col.write(mean_absolute_error(y, prediction))

    display_col.subheader('Mean squared error of the model is:')
    display_col.write(mean_squared_error(y, prediction))

    display_col.subheader('R squared score of the model is:')
    display_col.write(r2_score(y, prediction))
