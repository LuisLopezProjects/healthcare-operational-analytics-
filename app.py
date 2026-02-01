# app.py  # File name: main entry point for the Streamlit app
# Healthcare Operational Analytics | Streamlit Portfolio App  # Project title/purpose header
import streamlit as st  # Import Streamlit to build the web app UI
import pandas as pd  # Import pandas for reading and handling tabular data
import matplotlib.pyplot as plt  # Import matplotlib for plotting charts

DATA_PATH = "data/healthcare_dataset_clean.csv"  # Path to the cleaned dataset (Phase 2 output)

st.set_page_config(  # Configure the Streamlit page settings
    page_title="Healthcare Operational Analytics",  # Browser tab title
    page_icon="🏥",  # Emoji icon shown in the tab
    layout="wide"  # Use full-width layout for dashboards
)

@st.cache_data  # Cache the function output to avoid reloading the CSV on every interaction
def load_data(path: str) -> pd.DataFrame:  # Define a function that loads data and returns a DataFrame
    df = pd.read_csv(path)  # Read the CSV file into a pandas DataFrame
    return df  # Return the loaded DataFrame

########
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:  # Define a function that builds sidebar filters and returns filtered data
    st.sidebar.header("Filters")  # Add a title/header to the sidebar section

    condition_options = sorted(df["Medical Condition"].dropna().unique())  # Get sorted unique non-null medical conditions
    admission_options = sorted(df["Admission Type"].dropna().unique())  # Get sorted unique non-null admission types
    age_group_options = sorted(df["Age_Group"].dropna().unique())  # Get sorted unique non-null age groups

    selected_conditions = st.sidebar.multiselect(  # Create a multiselect widget for medical condition
        "Medical Condition",  # Label shown to the user
        options=condition_options,  # Options available for selection
        default=condition_options  # Default selection (all options selected)
    )

    selected_admission = st.sidebar.multiselect(  # Create a multiselect widget for admission type
        "Admission Type",  # Label shown to the user
        options=admission_options,  # Options available for selection
        default=admission_options  # Default selection (all options selected)
    )

    selected_age_groups = st.sidebar.multiselect(  # Create a multiselect widget for age group
        "Age Group",  # Label shown to the user
        options=age_group_options,  # Options available for selection
        default=age_group_options  # Default selection (all options selected)
    )

    filtered = df[  # Start boolean filtering of the DataFrame
        df["Medical Condition"].isin(selected_conditions)  # Keep rows whose condition is in the selected conditions
        & df["Admission Type"].isin(selected_admission)  # AND keep rows whose admission type is selected
        & df["Age_Group"].isin(selected_age_groups)  # AND keep rows whose age group is selected
    ].copy()  # Copy the filtered result to avoid chained-assignment issues

    return filtered  # Return the filtered DataFrame
#####

def show_kpis(df: pd.DataFrame) -> None:  # Define a function that displays KPI(Key Performance Indicator) metrics
    col1, col2, col3, col4 = st.columns(4)  # Create 4 columns for KPI(Key Performance Indicator) cards

    with col1:  # Place the first KPI inside column 1
        st.metric("Patients", f"{len(df):,}")  # Display the number of rows (patients) with comma formatting

    with col2:  # Place the second KPI inside column 2
        st.metric("Avg Length of Stay", f"{df['Length_of_Stay'].mean():.2f} days")  # Display mean LOS with 2 decimals

    with col3:  # Place the third KPI inside column 3
        st.metric("Avg Billing Amount", f"${df['Billing Amount'].mean():,.2f}")  # Display mean billing with currency formatting

    with col4:  # Place the fourth KPI inside column 4
        abnormal_rate = (df["Test Results"].eq("Abnormal").mean() * 100)  # Compute % of rows with "Abnormal" test result
        st.metric("Abnormal Tests", f"{abnormal_rate:.1f}%")  # Display abnormal rate with 1 decimal percent
################################################
def chart_los_distribution(df: pd.DataFrame) -> None:  # Define a function to plot LOS distribution
    st.subheader("Length of Stay Distribution")  # Add a section title in the main page

    fig, ax = plt.subplots(figsize=(8, 4))  # Create a figure and axis with a specific size
    ax.hist(df["Length_of_Stay"], bins=30)  # Plot a histogram of LOS with 30 bins
    ax.set_xlabel("Length of Stay (days)")  # Set x-axis label
    ax.set_ylabel("Number of Patients")  # Set y-axis label

    st.pyplot(fig)  # Render the matplotlib figure inside the Streamlit app
#####################################################
def chart_los_vs_billing(df: pd.DataFrame) -> None:
    """
    Scatter plot to explore the relationship between Length of Stay and Billing Amount.
    This helps identify whether longer stays tend to generate higher costs (a potential cost driver).
    """
    st.subheader("Cost Drivers: Length of Stay vs Billing Amount")

    # Defensive check: if filters return no rows, don't try to plot
    if df.empty:
        st.info("No records match the selected filters. Please broaden your selection.")
        return

    # Optional defensive check: very small samples can look misleading
    if len(df) < 5:
        st.warning("Very small sample size (fewer than 5 records). Interpretation may be unstable.")
        st.dataframe(df[["Length_of_Stay", "Billing Amount"]], use_container_width=True)
        return
    # Create the figure for matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))

    # Scatter plot: X = Length of Stay, Y = Billing Amount
    ax.scatter(df["Length_of_Stay"], df["Billing Amount"], alpha=0.4)

    # Titles and labels for interpretability
    ax.set_xlabel("Length of Stay (days)")
    ax.set_ylabel("Billing Amount ($)")
    ax.set_title("Longer stays may correlate with higher billing (check clustering and outliers)")

    # Render the figure in Streamlit
    st.pyplot(fig)

    # Quick interpretation block (executive-friendly)
    st.caption(
        "Interpretation: Look for upward trends (positive correlation) and clusters by segment. "
        "Outliers may indicate unusual billing cases worth auditing."
    )
#######################################################
def chart_avg_billing_by_condition(df: pd.DataFrame) -> None:
    # Show a section header in the Streamlit app
    st.subheader("Average Billing Amount by Medical Condition")

    # If the filtered DataFrame is empty, show an info message and stop the function
    if df.empty:
        st.info("No records match the selected filters. Please broaden your selection.")
        return

    # Group the data by Medical Condition and compute the average Billing Amount
    avg_billing = df.groupby("Medical Condition")["Billing Amount"].mean()

    # Sort the average billing values from highest to lowest for clearer ranking
    avg_billing = avg_billing.sort_values(ascending=False)

    # T1 Show the computed averages as a table for transparency
    st.dataframe(avg_billing.reset_index().rename(columns={"Billing Amount": "Avg Billing"}))

    # T2 Compute the minimum average billing across conditions
    min_avg = float(avg_billing.min())

    # Compute the maximum average billing across conditions
    max_avg = float(avg_billing.max())

    # Compute the percentage spread between max and min (relative variability)
    spread_pct = ((max_avg - min_avg) / max_avg) * 100 if max_avg != 0 else 0

    # Display these quick validation metrics
    st.write(f"Min average billing: ${min_avg:,.2f}")
    st.write(f"Max average billing: ${max_avg:,.2f}")
    st.write(f"Spread across conditions: {spread_pct:.2f}%")

    # =========================
    # Test 3: Relative Variability (CV – Coefficient of Variation)
    # =========================
    # Import numpy for numerical calculations
    import numpy as np

    # Convert average billing values to a NumPy array
    avg_values = avg_billing.values

    # Compute the mean of the average billing amounts
    mean_avg = float(np.mean(avg_values))

    # Compute the standard deviation of the average billing amounts
    std_avg = float(np.std(avg_values))

    # Defensive check: avoid division by zero
    if mean_avg == 0:
        cv_pct = 0
    else:
        # Compute Coefficient of Variation (relative dispersion)
        cv_pct = (std_avg / mean_avg) * 100

    # Display the CV result
    st.write(f"Coefficient of Variation (CV): {cv_pct:.2f}%")

    # Interpret the CV result for executive understanding
    if cv_pct < 5:
        st.success(
            "Interpretation: Average billing is highly uniform across medical conditions. "
            "Differences are statistically negligible."
        )
    elif cv_pct < 10:
        st.warning(
            "Interpretation: Moderate variability detected. Some condition-based differences may exist."
        )
    else:
        st.error(
            "Interpretation: High variability detected. Medical condition strongly impacts billing."
        )

    # Create a matplotlib figure and axis for the bar chart
    fig, ax = plt.subplots(figsize=(9, 4))

    # Plot the bar chart: x = condition names, y = average billing amount
    ax.bar(avg_billing.index, avg_billing.values)

    # Label the x-axis to indicate the categories
    ax.set_xlabel("Medical Condition")

    # Label the y-axis to indicate the metric being measured
    ax.set_ylabel("Average Billing Amount ($)")

    # Add a chart title to make the plot self-explanatory
    ax.set_title("Average Cost per Medical Condition")

    # Rotate x-axis labels so they do not overlap
    plt.xticks(rotation=45, ha="right")

    # Improve spacing so labels are not cut off
    plt.tight_layout()

    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)

    # Add a short executive-friendly interpretation below the chart
    st.caption(
        "Interpretation: Conditions with higher average billing may be key cost drivers "
        "and candidates for deeper investigation or operational optimization."
    )

#######################################################
st.title(" Healthcare Operational Analytics")  # Main title at the top of the dashboard

st.markdown(  # Add a markdown description below the title
    """
    This dashboard explores **healthcare operational and financial patterns**
    to identify **cost drivers**, **length-of-stay behavior**, and
    **executive-level insights**.
    """
)

df = load_data(DATA_PATH)  # Load the full dataset once (cached after the first run)
df_filtered = apply_filters(df)  # Apply user-selected filters and get a filtered dataset

st.subheader("Dataset Overview (Filtered)")  # Add a section title for the filtered overview
show_kpis(df_filtered)  # Display KPI cards based on the filtered data

st.subheader("Sample Records")  # Add a section title for sample rows
st.dataframe(df_filtered.head(10), use_container_width=True)  # Show the first 10 rows in a responsive table

chart_los_distribution(df_filtered)  # Plot the LOS distribution chart based on the filtered data

chart_los_vs_billing(df_filtered)

chart_avg_billing_by_condition(df_filtered)

st.markdown("---")  # Add a horizontal divider line
st.caption("Portfolio project | Streamlit + Pandas + Matplotlib")  # Add a small footer caption

