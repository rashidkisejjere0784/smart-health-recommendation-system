import data_cleaning as data_cleaner
import similarity
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import gmaps

data = pd.read_excel("Hospital Data.xlsx")

# Initialize necessary objects
DataCleaner = data_cleaner.DataCleaner()
st.set_page_config(layout="wide")


def display_hospital_info(hospital):
    with st.expander(f"## {hospital['Hospital']}"):
        st.write(f"### {hospital['Location']}")
        st.write(f"**Services Offered:** {hospital['cleaned services']}")
        st.write(f"**Payment Options:** {hospital['payment']}")
        st.write(f"**Rating:** {hospital['rating']}")

        # Show website as a clickable link
        st.write(f"**Website:** [{hospital['website']}]({hospital['website']})")

        st.subheader("Opening Hours")
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_of_week:
            st.write(f"**{day}:** {hospital[day]}")

def reload_page():
    # Set a session state flag to trigger rerun
    st.session_state.reload_flag = True

    # Immediately rerun the app from the top
    st.rerun()



# Main Streamlit app
def main():
    st.title("Smart Health Care Recommendation System")

    # User input section
    column_1, column_2 = st.columns([1, 1])
    with column_1:
        global services, payment_system, care_system, rating, Location
        st.subheader("Select Your Features")
        subcol1, subcol2 = st.columns([3, 1])
        with subcol1:
            global services
            services = st.text_input("Services", help="Which services are you looking for", on_change=get_recommendation)
        with subcol2:
            global n
            n = st.selectbox("n", options=[10, 15, 20, 15], on_change=get_recommendation)

        payment_system = st.selectbox("Payment System", options=[None, "insurance", "No Payment", "cash"])
        care_system = st.selectbox("Care System", options=[None, "Public", "Private"])
        Location = st.selectbox("Location", options=[None, "Kampala", "Gulu", "Jinja", "Mukono"])

        with st.expander("More features"):
            st.header("Select the Open Hours")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                global Monday,Tuesday, Wednesday
                Monday = st.selectbox("Monday", options=[None, 'Open 24 hours'])
                Tuesday = st.selectbox("Tuesday", options=[None, 'Open 24 hours'])
                Wednesday = st.selectbox("Wednesday", options=[None, 'Open 24 hours'])

            with col2:
                global Thursday, Friday, Saturday, Sunday
                Thursday = st.selectbox("Thursday", options=[None, 'Open 24 hours'])
                Friday = st.selectbox("Friday", options=[None, 'Open 24 hours'])
                Saturday = st.selectbox("Saturday", options=[None, 'Open 24 hours'])

            with col3:
                global Sunday
                Sunday = st.selectbox("Sunday", options=[None, 'Open 24 hours'])

            global rating
            rating = st.slider("Rating", min_value=0, max_value=5)
            

        
        get_recommendation_button = st.button("Get Recommendation", on_click=get_recommendation)

    with column_2:

        st.header("Recommended")
        if 'hospital_indicies' in st.session_state:
            hospitals = data.iloc[st.session_state['hospital_indicies']]
            for index in range(len(hospitals)):
                hospital = hospitals.iloc[index]
                display_hospital_info(hospital)


def get_recommendation():
    vectors, full_encoded_data = DataCleaner.get_vector_matrices(services=services,
                                                                 Location=Location, Monday=Monday, Tuesday=Tuesday, Wednesday=Wednesday,
                                                                 Thursday=Thursday, Friday=Friday, Saturday=Saturday,
                                                                 Sunday=Sunday, rating=rating,
                                                                 payment=payment_system,
                                                                 care_system=care_system
                                                                 )
    
    hospital_indicies = similarity.get_grounded_predictions(vectors, full_encoded_data, n)
    st.session_state['hospital_indicies'] = hospital_indicies[::-1]
    print(hospital_indicies)
    reload_page()


if __name__ == "__main__":
    main()