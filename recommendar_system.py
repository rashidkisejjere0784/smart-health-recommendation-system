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
    with st.expander(f"{hospital['Hospital']} ({hospital['Location']})"):
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
    st.experimental_rerun()

# # Function to render the map
# def render_map(marker_locations):
#     st.subheader("Map View")

#     # Create a map figure centered at initial coordinates
#     initial_coordinates = (40.75, -74)
#     fig = gmaps.figure(center=initial_coordinates, zoom_level=12)

#     # Create a list of marker locations based on user input
#     markers = gmaps.marker_layer(marker_locations)
#     fig.add_layer(markers)

#     # Display the map using components.html()
#     components.html(fig)    
#     st.subheader("Map View")

#     # Create a map figure centered at initial coordinates
#     initial_coordinates = (40.75, -74)
#     map_figure = gmaps.figure(center=initial_coordinates, zoom_level=12)

#     # Define locations for placemarks (markers)
#     marker_locations = [
#         (40.75, -74),  # Example marker 1 (latitude, longitude)
#         (40.8, -74.1),  # Example marker 2 (latitude, longitude)
#         (40.7, -74.2)   # Example marker 3 (latitude, longitude)
#     ]

#     # Add markers to the map
#     markers = [gmaps.marker(location=loc) for loc in marker_locations]
#     marker_layer = gmaps.marker_layer(markers)
#     map_figure.add_layer(marker_layer)

#     # Display the map using components.html()
#     components.html(map_figure)


# Main Streamlit app
def main():
    st.title("Smart Health Care Recommendation System")

    # User input section
    col1, col2 = st.columns([1, 1])
    with col1:
        global services, payment_system, care_system, latitude, longitude, rating
        st.subheader("Select Your Features")
        subcol1, subcol2 = st.columns([3, 1])
        with subcol1:
            global services
            services = st.text_input("Services", help="Which services are you looking for", on_change=get_recommendation)
        with subcol2:
            global n
            n = st.selectbox("n", options=[3, 5, 10, 15], on_change=get_recommendation)

        payment_system = st.selectbox("Payment System", options=["None", "Insurance", "No Payment"])
        care_system = st.selectbox("Care System", options=["None", "Public", "Private"])
        latitude = st.number_input("Latitude", min_value=-90, max_value=90)
        longitude = st.number_input("Longitude", min_value=-180, max_value=180)
        rating = st.slider("Rating", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
        get_recommendation_button = st.button("Get Recommendation", on_click=get_recommendation)

    with col2:
        st.header("Recommended")
        if 'hospital_indicies' in st.session_state:
            hospitals = data.iloc[st.session_state['hospital_indicies']]
            
            for index in range(len(hospitals)):
                hospital = hospitals.iloc[index]
                col2.empty()
                display_hospital_info(hospital)


def get_recommendation():
    global rating, latitude, longitude, payment_system, care_system, n
    # Placeholder for recommendation logic
    if rating == "None":
        rating = None

    if latitude == "None":
        latitude = None

    if longitude == "None":
        longitude = None

    if payment_system == "None":
        payment_system = None

    if care_system == "None":
        care_system = None

    vectors, full_encoded_data = DataCleaner.get_vector_matrices(services,
                                                                latitude,
                                                                longitude,rating,
                                                                care_system,payment_system)
    
    hospital_indicies = similarity.calculate_cosine_similarity(vectors, full_encoded_data, n)
    st.session_state['hospital_indicies'] = hospital_indicies

    reload_page()

if __name__ == "__main__":
    main()