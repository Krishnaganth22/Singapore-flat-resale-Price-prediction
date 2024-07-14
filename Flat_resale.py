
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import bz2

# Setting up the app layout
st.set_page_config(layout='wide')
st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Singapore Flat Resale</h1>
</div>
""", unsafe_allow_html=True)
# Giving the option menu About project and Prediction options
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "rocket"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
if selected=='About Project':
    st.markdown("# :red[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown('### :blue[Overview :] Introduction Welcome to the Singapore Flat Resale Price Prediction project! This project aims to leverage machine learning techniques to predict the resale prices of flats in Singapore. The resale prices of flats are influenced by a variety of factors including location, flat type, block, street name, storey range, floor area, flat model, lease commencement date, year, and month. By building a predictive model, we aim to provide valuable insights and accurate price predictions for buyers and sellers in the real estate market.')
    

                
    st.markdown("### :blue[Domain :] Real Estate")
#In prediction menu
if selected=='Predictions':
    st.markdown("# :red[Predicting results based on Trained models]")
    st.markdown('# :blue[Predicting resale Price by (Decisiontree Regressor) with (Accuracy: 96%)]')
    try:
        # Loaded the data and assign variables for encoded data
        file=pd.read_csv('Flat_sing.csv.gz',low_memory=False)
        data=pd.DataFrame(file) 
        
        # Load the trained model and encoders
      
        with bz2.BZ2File('krish.pkl.bz2', 'rb') as file:
            model = pickle.load(file)

        # Assuming the encoders are saved separately, load them
        with open('label_encoder_street_name.pkl', 'rb') as file:
            label_encoder_street_name = pickle.load(file)
        with open('label_encoder_flat_type.pkl', 'rb') as file:
            label_encoder_flat_type = pickle.load(file)
        with open('label_encoder_flat_model.pkl', 'rb') as file:
            label_encoder_flat_model = pickle.load(file)
        with open('label_encoder_town.pkl', 'rb') as file:
            label_encoder_town = pickle.load(file)
        with open('label_encoder_storey_range.pkl', 'rb') as file:
            label_encoder_storey_range = pickle.load(file)
        
        #Converting block column into str

        data['block']=data['block'].astype('str')
        
        unique_block=data['block'].unique()

        def user_input_features():
            town = st.selectbox('Town', label_encoder_town.classes_)
            flat_type = st.selectbox('Flat Type', label_encoder_flat_type.classes_)
            block = st.selectbox('Block',unique_block)
            street_name = st.selectbox('Street Name', label_encoder_street_name.classes_)
            storey_range = st.selectbox('Storey Range', label_encoder_storey_range.classes_)
            floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0.0, step=1.0)
            flat_model = st.selectbox('Flat Model', label_encoder_flat_model.classes_)
            lease_commence_date = st.number_input('Lease Commence Date', min_value=1900, max_value=2100, step=1)
            year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
            month = st.number_input('Month', min_value=1, max_value=12, step=1)
            
            # Encode the categorical features using the saved LabelEncoders
            street_name_encoded = label_encoder_street_name.transform([street_name])[0]
            flat_type_encoded = label_encoder_flat_type.transform([flat_type])[0]
            flat_model_encoded = label_encoder_flat_model.transform([flat_model])[0]
            town_encoded = label_encoder_town.transform([town])[0]
            storey_range_encoded = label_encoder_storey_range.transform([storey_range])[0]
            # Loaded the features into data and returen features
            data = {
                'town': town_encoded,
                'flat_type': flat_type_encoded,
                'block': block,
                'street_name': street_name_encoded,
                'storey_range': storey_range_encoded,
                'floor_area_sqm': floor_area_sqm,
                'flat_model': flat_model_encoded,
                'lease_commence_date': lease_commence_date,
                'Year': year,
                'Month': month
            }
            
            features = pd.DataFrame(data, index=[0])
            return features

        def main():
            st.title('Singapore flat Resale Price Prediction')
            st.write("Enter the details of the flat to predict its resale price:")

            input_df = user_input_features()

            if st.button('Predict'):
                # Ensure the block column is correctly typed
                input_df['block'] = input_df['block'].apply(lambda x: ''.join(char for char in x if char.isdigit()))
                # Predict the price using user input
                prediction = model.predict(input_df)
                st.write(f'The predicted resale price is: ${prediction[0]:,.2f}')

        if __name__ == '__main__':
            main()
        

            
    
    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")


    
    
