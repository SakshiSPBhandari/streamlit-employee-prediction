import streamlit as st
import numpy as np
from PIL import Image
image1 = Image.open('leave1.jpg')
image2 = Image.open('positive.jpg')
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))

df = pickle.load(open('df.pkl','rb'))




st.sidebar.header("Select below options")

#education
education = st.sidebar.selectbox('Education',df['Education'].unique())

#joining year
year = st.sidebar.selectbox('Year',
df['JoiningYear'].unique())

# City
city = st.sidebar.selectbox('City',df['City'].unique())

#tier
tier = st.sidebar.selectbox('PaymentTier',[1,2,3])

#age
age = st.sidebar.number_input("Age")

#gender
gender = st.sidebar.selectbox("Gender",
df['Gender'].unique())

#EvenBenched
benched = st.sidebar.selectbox("Wasn't part of project in 2 months",df['EverBenched'].unique())

#experience
exp = st.sidebar.selectbox("Experience",[0,1,2,3,4,5,6,7])


if st.button('Predict Leave or Not'):
    query = np.array([education,year,city,tier,age,gender,benched,exp],dtype=object).reshape(1,8)

    ans = int(np.exp(pipe.predict(query)[0]))

    if ans==1 :
        st.header("Employee will leave")
        st.image(image1,width=500)
    else:
        st.header("Employee will not leave")
        st.image(image2,width=500)
    # st.title("The prediction of the employee leave or not is "+str(int(np.exp(pipe.predict(query)[0]))))