import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
from streamlit_option_menu import option_menu
import re
import pickle

def get_df():
    df = pd.read_csv("copper.csv").reset_index(drop= True)
    return df
            
def load_pickle_regg(quantity_tons,thickness,width,customer):
    flag = 0 
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in [quantity_tons,thickness,width,customer]:             
        if re.match(pattern, i):
            pass
        else:                    
            flag = 1  
            break

    if flag == 1:
        if len(i) == 0:
            st.warning("Please Enter a Valid number space not allowed")
        else:
            st.write("You have Entered an Invalid Value: ",i)  

    elif flag == 0:
        with open(r"model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open(r'scaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"t.pkl", 'rb') as f:
            t_loaded = pickle.load(f)

        with open(r"s.pkl", 'rb') as f:
            s_loaded = pickle.load(f)

        return loaded_model, scaler_loaded, t_loaded, s_loaded
    
def load_pickle_class(cquantity_tons,cthickness,cwidth,ccustomer,cselling):
    cflag=0 
    pattern = "^(?:\d+|\d*\.\d+)$"
    for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
        if re.match(pattern, k):
            pass
        else:                    
            cflag=1  
            break

    if cflag == 1:
        if len(k) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ",k)  

    elif cflag == 0:
        import pickle
        with open(r"cmodel.pkl", 'rb') as file:
            cloaded_model = pickle.load(file)

        with open(r'cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)
        return cloaded_model, cscaler_loaded, ct_loaded

df = get_df()

# Setting Page config
st.set_page_config(
    page_title="Industrial Copper Modeling",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded")

#Title
st.write("""

<div style='text-align:center'>
    <h1 style='color:#ff6600;'>Industrial Copper Modeling</h1>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["HOME","PREDICT SELLING PRICE", "PREDICT STATUS"]) 

with tab1:
  
        h4 = " "
st.write("#")
st.write(f'<h4 style="color:lightblue;">This project focuses on developing two machine learning models for the copper industry, aimed at predicting the selling price and classifying leads effectively.</h4>', unsafe_allow_html=True)
st.write(f'<h4 style="color:lightblue;">Traditional manual predictions are often time-consuming and may not lead to optimal pricing decisions or accurate lead classification.</h4>', unsafe_allow_html=True)
st.write(f'<h4 style="color:lightblue;">Our models will employ advanced techniques such as data normalization, outlier detection and handling, correction of improperly formatted data, feature distribution analysis, and the use of tree-based models like decision trees to ensure accurate predictions of selling prices and lead classifications.</h4>', unsafe_allow_html=True)
st.write("    ")
st.write("### :orange[DOMAIN:] MANUFACTURING")
st.write("""
                ### :orange[TECHNOLOGIES USED:]

                #### :green[PYTHON]
                #### :green[PANDAS]
                #### :green[DATA PREPROCESSING]
                #### :green[EDA]
                #### :green[SCIKIT-LEARN]
                #### :green[STREAMLIT]
            """)
       

with tab2:    

        # Define the possible values for the dropdown menus
        status_options = df['status'].value_counts().keys().tolist()
        item_type_options = df['item type'].value_counts().keys().tolist()
        country_options = df['country'].value_counts().keys().tolist()
        application_options = df['application'].value_counts().keys().tolist()
        product = df['product_ref'].value_counts().keys().tolist()

        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col3:               
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                    background-color: #0073e6;
                    color: white;
                    width: 100%;
                    border-radius: 10px;
                    border: 2px solid #004080;
                    padding: 10px;
                    font-size: 18px;
                     }
                    </style>
                """, unsafe_allow_html=True)
    
                
                if submit_button:
                    loaded_model,scaler_loaded,t_loaded,s_loaded = load_pickle_regg(quantity_tons,thickness,width,customer)

                    new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
                    new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
                    new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
                    new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
                    new_sample1 = scaler_loaded.transform(new_sample)
                    new_pred = loaded_model.predict(new_sample1)[0]
                    st.write(f'<h2 class="fancy-text">Predicted selling price: {predicted_price}</h2>', unsafe_allow_html=True)

with tab3: 
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            
           
            if csubmit_button:
                cloaded_model,cscaler_loaded,ct_loaded = load_pickle_class(cquantity_tons,cthickness,cwidth,ccustomer,cselling)

            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
                new_sample = cscaler_loaded.transform(new_sample)
                new_pred = cloaded_model.predict(new_sample)
                if new_pred==1:
                    st.write('## :green[The Status is Won 🏆] ')
                else:
                    st.write('## :red[The Status is Lost 😞] ')

