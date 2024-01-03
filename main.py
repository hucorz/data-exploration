import streamlit as st
from st_ant_tree import st_ant_tree
import os
import pandas as pd
import numpy as np

from components import utils

import openai
from openai import OpenAI
import json
from PIL import Image


# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="DEMO",
    page_icon="📊",
)

st.write("# DEMO 📊")

st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key and base url
with open('./openai_setup.json', 'r') as f:
    openai_setup = json.load(f)
    openai_key = openai_setup["openai_key"]
    openai_base_url = openai_setup["openai_base_url"]
# openai_base_url = os.environ.get("OPENAI_BASE_URL")
print("===========",openai_key,openai_base_url)

if not openai_key or not openai_base_url:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    openai_base_url = st.sidebar.text_input("Enter OpenAI base url:", value="default")
    if openai_key and openai_base_url:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(f"Current key: {display_key}")
        st.sidebar.write(f"Current base url: {openai_base_url}")
else:
    display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
    display_base_url = openai_base_url
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")
    st.sidebar.write(
        f"OpenAI base url loaded from environment variable: {display_base_url}"
    )

st.markdown(
    """
    Description: TODO.

   ----
"""
)

if openai_base_url == "default":
    openai_base_url = "https://api.openai.com/v1"

finish_setup = openai_key and openai_base_url

# Step 2 - Select a dataset and summarization method
if finish_setup:
    client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_key,
    )

    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox("Choose a model", options=models, index=0)

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.0
    )

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    st.sidebar.write("### Choose a dataset")

    datasets = [
        # {"label": "Select a dataset", "url": None},
        {
            "label": "Cars",
            "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv",
        },
        {
            "label": "Weather",
            "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json",
        },
    ]

    selected_dataset_label = st.sidebar.selectbox(
        "Choose a dataset", options=[dataset["label"] for dataset in datasets], index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or JSON file", type=["csv", "json"]
        )

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[
            [dataset["label"] for dataset in datasets].index(selected_dataset_label)
        ]["url"]

    if not selected_dataset:
        st.info(
            "To continue, select a dataset from the sidebar on the left or upload your own."
        )

    st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {
            "label": "default",
            "description": "Uses dataset column statistics and column names as the summary",
        },
        {
            "label": "columns",
            "description": "Uses the dataset column names as the summary",
        },
        {
            "label": "llm",
            "description": "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description",
        },
    ]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        "Choose a method",
        options=[method["label"] for method in summarization_methods],
        index=0,
    )

    selected_method = summarization_methods[
        [method["label"] for method in summarization_methods].index(
            selected_method_label
        )
    ]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[
        [method["label"] for method in summarization_methods].index(
            selected_method_label
        )
    ]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True,
        )

# Step 3 - Generate data summary
if finish_setup and selected_dataset and selected_method:
    st.write("## Summary")
    # **** lida.summarize *****
    summary = utils.get_data_summary('cars',selected_dataset, summary_method=selected_method)

    print(summary)

    field_cnt = len(summary["fields"])

    shape = (field_cnt + 2 // 3, 3)
    col_list = [st.columns(3) for i in range(shape[0])]

    for idx in range(field_cnt):
        row = idx // shape[0]
        col = idx % shape[1]
        with col_list[row][col]:
            variables = {}
            st.header(summary["fields"][idx]["column"])
            # exec(summary["fields"][idx]["properties"]["code"], variables)
            # st.pyplot(variables["fig"])

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])
    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    if "Correlation" in summary:
        Correlations = summary["Correlation"]
        nfields_df = pd.DataFrame(Correlations)
        st.write(nfields_df)

    # Step 4 - Start exploring the data
    st.write("## Explore")

    openai_client = OpenAI(
        base_url=openai_base_url,
        api_key=openai_key
    )

    simple_facts = utils.get_fact_from_data_summary(openai_client,summary)
    st.write(simple_facts)

    st.write("## User selection facts")
    user_select_simple_facts = [
        "The 'Type' field categorizes the cars into 5 different categories, with 'Sedan' being the most commonly occurring type. This points to Sedan as the dominant car type in the data represented.",
        # "The average retail price of the cars in the dataset is approximately $33,253.2, with a standard deviation of $19,766. This sizable standard deviation clearly accentuates the wide price range of the car market captured in the dataset, extending from as low as $10,280 up to an outlier price of $192,465.",
        # "The relationship between engine size and weight of the cars is quite strong, with a Pearson's correlation of approximately 0.8. This typifies a typical trend in vehicle manufacturing, where larger engines typically contribute to a heavier overall vehicle weight.",
        # "It is noticeable that there's a significant negative correlation (approximately -0.79) between 'Highway_Miles_Per_Gallon' and 'Weight', indicating that heavier cars tend to have a lower miles per gallon efficiency on highways."
    ]
    st.write(user_select_simple_facts)

    st.write("## chart-type")

    for user_select_simple_fact in user_select_simple_facts:
        chart_type = utils.get_chart_type_from_fact(openai_client,user_select_simple_fact,summary)
        st.write(chart_type["fact"])
        st.write(chart_type["data field"])
        st.write(chart_type["visualization"])
        st.write(chart_type["rationale"])
    
    st.write("## visualization")
    generated_fact_code = utils.get_chart_code(openai_client,chart_type["visualization"],chart_type["fact"],summary)
    st.write(generated_fact_code)
    # 根据代码生成图表
    chart_file_path = utils.get_chart_file_path(generated_fact_code,f"{'-'.join(chart_type['data field'])}-{chart_type['visualization']}")
    # 已有图表，显示在网页上
    image = Image.open(chart_file_path)
    st.image(image, caption=chart_type["fact"], use_column_width=True)


    tree_data = [
        {
            "value": "parent 1",
            "title": "Parent 1",
            "children": [
                {"value": "child 1", "title": "Child 1"},
                {"value": "child 2", "title": "Child 2"},
            ],
        },
        {
            "value": "parent 2",
            "title": "Parent 2",
        },
    ]
    selected_node = st_ant_tree(
        tree_data, treeCheckable=False, treeDefaultExpandAll=True
    )
    print(selected_node)
