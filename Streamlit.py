import streamlit as st
import pandas as pd
from datetime import date

# 1. Initialize the dataframe in session state if it doesn't exist
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "dt1": [date.today()],
        "dt2": [date.today()],
        "dt3": [date.today()]
    })

st.title("Editable Date Table")

# 2. Display the data editor
# num_rows="dynamic" adds the "+" button at the bottom of the table
edited_df = st.data_editor(
    st.session_state.df,
    column_config={
        "dt1": st.column_config.DateColumn("Date 1", required=True),
        "dt2": st.column_config.DateColumn("Date 2", required=True),
        "dt3": st.column_config.DateColumn("Date 3", required=True),
    },
    num_rows="dynamic",
    use_container_width=True,
    key="date_editor"
)

# 3. Update the session state with the edited data
if st.button("Save Changes"):
    st.session_state.df = edited_df
    st.success("Dataframe updated!")

st.write("Current Dataframe:", st.session_state.df)
