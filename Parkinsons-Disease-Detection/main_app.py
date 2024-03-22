import streamlit as st
import pandas as pd
import app

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    # Create Tabs
    tab1, tab2, tab3 = st.columns(3)
    with tab1:
        st.markdown("<h1 style='text-align: center; color: #ff5733;'>Home Page</h1>", unsafe_allow_html=True)
        st.write("\n")
        st.markdown("<h2 style='text-align: center; color: #3366ff;'>Welcome to the information desk.</h2>", unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")
        st.markdown("<h3 style='text-align: center; color: #33cc33;'>Instructions:</h3>", unsafe_allow_html=True)
        st.write("\n")
        st.write("<ul><li style='color: #3333ff;'>Visit Home page for instructions.</li>"
                 "<li style='color: #3333ff;'>Visit 'Test Report and Update' page for generating report.</li>"
                 "<li style='color: #3333ff;'>Visit 'Individual Patient Update' page for collecting individual report.</li></ul>",
                 unsafe_allow_html=True)

    with tab2:
        st.markdown("<h1 style='text-align: center; color: #ff5733;'>Test Report and Update</h1>", unsafe_allow_html=True)
        st.write("\n")
        st.write("<p style='text-align: center; color: #3333ff;'>Kindly, upload all the complete patient report here.</p>", unsafe_allow_html=True)

        df_file = st.file_uploader('Upload a CSV', type=['csv'], help='Only CSV file is acceptable')

        if df_file is not None:
            df = pd.read_csv(df_file)
            st.dataframe(df)
            patient_prediction , full_csv = app.run_app(df)
            csv = convert_df(full_csv)
            st.download_button('Download updated file as CSV', data=csv, file_name='updated_file.csv', mime='text/csv')

        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("<p style='color: #3333ff;'>Patient report should have: patient ID, MDVP:Fo (Hz), MDVP:Fhi (Hz), MDVP:Flo (Hz), MDVP:Jitter (%), MDVP:Jitter (Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer (dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, and PPE.</p>",
                 unsafe_allow_html=True)

    with tab3:
        st.markdown("<h1 style='text-align: center; color: #ff5733;'>Individual Patient Update</h1>", unsafe_allow_html=True)
        st.write("\n")
        st.write("<p style='text-align: center; color: #3333ff;'>If you want any individual patient's details kindly input his/her ID below.</p>", unsafe_allow_html=True)

        with st.form(key='patient_form', clear_on_submit = True):
            patient_id = st.text_input('Patient ID') 
            patient_name = st.text_input('Patient Name')
            submitted_button = st.form_submit_button('Submit')

        if submitted_button:
            if patient_id != "":
                    st.markdown("<h3 style='color: #33cc33;'>Details:</h3>", unsafe_allow_html=True)
                    st.write("Patient ID : ", patient_id)

                    if patient_name != "":
                        st.write("Patient Name : ", patient_name)

                    p_pred_value = patient_prediction.loc[patient_prediction['Patient ID'] == patient_id, 'Prediction'].item()

                    if patient_id in patient_prediction['Patient ID'].values :
                        st.write("Patient Report : ", p_pred_value)
                    else:
                        st.write("Patient Report : Not Available")
                              
            else:
                st.write("At least Patient ID is Required")

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if __name__ == "__main__":
    main()
