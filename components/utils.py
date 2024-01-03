from lida import Manager, TextGenerationConfig , llm
import json
import copy
import pandas as pd
import numpy as np
from collections import Counter
import math
from itertools import combinations
import textwrap

def get_recommand_field(*args, **kwargs):
    pass

def detect_outliers_iqr(data, max_outliers=6, max_multiplier = 3):
    multiplier = 1
    
    while True:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        
        iqr = q3 - q1
        
        lower_threshold = q1 - multiplier * iqr
        upper_threshold = q3 + multiplier * iqr
        
        outliers = np.where((data < lower_threshold) | (data > upper_threshold))[0]
        
        if len(outliers) <= max_outliers or multiplier >= max_multiplier:
            break
        
        multiplier += 0.1
    
    return outliers

def get_data_summary(data_name, *args, **kwargs):
    """
    Returns a summary of the dataset in the form of a dict
    """
    file_path = f"./data/{data_name}.csv"
    lida = Manager(text_gen = llm("openai", api_key="wwww")) # !! api key
    textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)
    print("-----------------------------------------------")
    summary = lida.summarize(file_path, summary_method="default", textgen_config=textgen_config)
    # 遍历 data fields，然后找到数据属性的列，然后进行quickinsight分析
    summary_copy = copy.deepcopy(summary)
    df = pd.read_csv(file_path)
    number_field = []
    # 添加列数
    summary_copy["number_cols"] = df.shape[0]
    for index,field in enumerate(summary["fields"]):
        del summary_copy["fields"][index]["properties"]["samples"]
        col_type = field["properties"]["dtype"]
        if col_type == "category":
            data = df[field["column"]].tolist()
            data_counts = Counter(data)
            total_count = len(data)
            data_percentages = {key: (count / total_count) * 100 for key, count in data_counts.items()}
            sorted_percentages = sorted(data_percentages.items(), key=lambda x: x[1], reverse=True)
            summary_copy["fields"][index]["properties"]["Outstanding No.1"] = sorted_percentages[0][0]
            summary_copy["fields"][index]["properties"]["Outstanding No.Last"] = sorted_percentages[-1][0]
            summary_copy["fields"][index]["properties"]["Outstanding Top 2"] = ", ".join([f"{item[0]}" for item in sorted_percentages[0:2]])
            data_frequencies = {key: count / total_count for key, count in data_counts.items()}
            entropy = -sum([freq * math.log2(freq) for freq in data_frequencies.values()])
            max_entropy = math.log2(len(data_counts))
            evenness_index = round(entropy / max_entropy,2)
            summary_copy["fields"][index]["properties"]["Evenness"] = evenness_index
            print(evenness_index)
        if col_type in ['number','int64','float64']:
            number_field.append(field["column"])
            data = np.array(df[field["column"]].tolist())
            outliers_indices = detect_outliers_iqr(data,max_outliers = 4,max_multiplier=5)
            summary_copy["fields"][index]["properties"]["Outlier"] = list(set(data[outliers_indices].tolist()))
            summary_copy["fields"][index]["properties"]["mean"] = round(data.mean(),2)
            summary_copy["fields"][index]["properties"]["range"] = [data.min(),data.max()]
            # summary_copy["fields"][index]["properties"]["min"] = data.min()
            # summary_copy["fields"][index]["properties"]["std"] = round(data.std(),2)
            print("离群值:", list(set(data[outliers_indices].tolist())))
        if col_type == "date":
        # if True:
            # 说明是时序数据，需要根据其他number类型的数据来分析
            for index,field in enumerate(summary["fields"]):
                if field["properties"]["dtype"] in ['number','int64','float64']:
                    x = np.arange(len(df[field["column"]].tolist()))
                    slope, intercept = np.polyfit(x, df[field["column"]].tolist(), 1)
                    if slope > 0:
                        summary_copy["fields"][index]["properties"]["trend"] = "upward trend"
                    elif slope < 0:
                        summary_copy["fields"][index]["properties"]["trend"] = "downward trend"
                    else:
                        summary_copy["fields"][index]["properties"]["trend"] = "stable trend"
    combinations_list = list(combinations(number_field, 2))
    print(len(combinations_list))
    # summary_copy["Correlation"] = []
    # for f1,f2 in combinations_list:
    #     correlation_matrix = np.corrcoef(df[f1].tolist(), df[f2].tolist())
    #     correlation_coefficient = correlation_matrix[0, 1]
    #     # print(correlation_coefficient)
    #     if abs(correlation_coefficient)>0.5:
    #         summary_copy["Correlation"].append(f"{f1}, {f2}, Pearson's Correlation = {correlation_coefficient}")
    
    return summary_copy

def get_chart_type_from_fact(openai_client,fact,data_summary):
    gen_vis_system_prompt = """
You are a helpful assistant highly skilled in recommending the most suitable VISUAL CHART TYPE for data facts. Given data facts, generate the most suitable chart type based on the TEMPLATE.
"""
    FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    {"fact": "", "data field": ["Type"], "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""
    user_prompt = f"""A data fact is "{fact}",it is BASED ON THE DATA SUMMARY below, \n\n .
        {data_summary} \n\n"""
    user_prompt += f"""\n  The generated chart types SHOULD BE FOCUSED ON THE ACCURACY  \n"""
    # response = openai_client.chat.completions.create(
    #     # model="gpt-4",  # 聊天模型的引擎
    #     model = "gpt-3.5-turbo",
    #     messages = [
    #             {"role": "system", "content": gen_vis_system_prompt},
    #             {"role": "assistant",
    #             "content":
    #             f"{user_prompt}\n\n. {FORMAT_INSTRUCTIONS} \n\n The output type is: \n "}]
    # )

    # # 提取生成的文本
    # chart_type = response.choices[0].message.content
    chart_type = """
    {"fact": "The 'Type' field categorizes the cars into 5 different categories, with 'Sedan' being the most commonly occurring type.",
     "data field": ["Type"],
     "visualization": "bar chart",
     "rationale": "A bar chart is suitable for comparing the frequency of different categories. In this case, we can use a bar chart to show the number of occurrences of each car type."}
"""
    print(chart_type)
    return json.loads(chart_type)
    return chart_type

def get_fact_from_data_summary(openai_client,data_summary):

    """
    Return:
        ["","",""]
    """

    FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A LIST OF FACTS. IT MUST USE THE FOLLOWING FORMAT:

```
[
"Alaska (AK) boasts the highest quantity of airports among all states, totaling an impressive 263 airfields. This extensive network of airports is a testament to the state's heavy dependence on air travel, a necessity driven by its expansive and challenging terrain characterized by vast wilderness and rugged landscapes.",
"The average (mean) latitude and longitude across all airports in the dataset are approximately 40.04°N and -98.62°W. This point can be considered a central geographical location in the context of this dataset."
]
```
THE OUTPUT SHOULD ONLY USE THE LIST FORMAT ABOVE.
"""
    user_prompt = f"""The facts should BE BASED ON THE DATA SUMMARY below, \n\n .
        {data_summary} \n\n"""
    user_prompt += f"""\n You need to analyze the data summary provided in depth and Select some important data from the above summary, express it as fact \n"""
    user_prompt += f"""\n The generated facts SHOULD BE FOCUSED ON THE DATA. \n"""

    SYSTEM_INSTRUCTIONS = """You are a an experienced data analyst who can generate a number of facts about data, when given a summary of the data. """

    # response = openai_client.chat.completions.create(
    #     # model="gpt-4",  # 聊天模型的引擎
    #     model = "gpt-3.5-turbo",
    #     messages = [
    #             {"role": "system", "content": SYSTEM_INSTRUCTIONS},
    #             {"role": "assistant",
    #             "content":
    #             f"{user_prompt}\n\n. {FORMAT_INSTRUCTIONS} \n\n The generated complex and insightful facts are: \n "}]
    # )

    # # 提取生成的文本
    # simple_insights = response.choices[0].message.content
    simple_insights = """
The average engine size of the cars in the dataset is 3.12 liters, with a range of 0.0 to 6.0 liters.
The average number of cylinders in the cars in the dataset is 5.74, with a range of 0 to 12.
The average horsepower of the cars in the dataset is 214.29, with a range of 73 to 493.
The average city miles per gallon of the cars in the dataset is 20.33, with a range of 10 to 60.
The average highway miles per gallon of the cars in the dataset is 27.29, with a range of 12 to 66.
The average weight of the cars in the dataset is 3,530.40 pounds, with a range of 1,850 to 6,400 pounds.
The average wheel base of the cars in the dataset is 106.91 inches, with a range of 0 to 130 inches.
The average length of the cars in the dataset is 184.97 inches, with a range of 143 to 221 inches.
The average width of the cars in the dataset is 71.08 inches, with a range of 2 to 81 inches.
"""
    # print(simple_insights)
    return simple_insights


def get_chart_code(openai_client,visualization,fact,data_summary,library = "matplotlib"):
    vis_system_prompt = """
    You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some code template, you complete the template to generate a visualization given the dataset and the goal described. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
    """
    
    general_instructions = f"If the solution requires a single value (e.g. max, min, median, first, last etc), ALWAYS add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F). If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column i) convert date fields to date types using data[''] = pd.to_datetime(data[<field>], errors='coerce'), ALWAYS use  errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[<field>])] iii) convert field to right time format for plotting.  ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed). Solve the task  carefully by completing ONLY the <imports> AND <stub> section. Given the dataset summary, the plot(data) method should generate a {library} chart ({visualization}) that addresses this insight: {fact}. DO NOT WRITE ANY CODE TO LOAD THE DATA. The data is already loaded and available in the variable data."

    matplotlib_instructions = f" {general_instructions} DO NOT include plt.show(). The plot method must return a matplotlib object (plt). Think step by step. \n"

    library_instructions = {
        "role": "assistant",
        "content": f"  {matplotlib_instructions}. Use BaseMap for charts that require a map. "}
    library_template = \
        f"""
    import matplotlib.pyplot as plt
    import pandas as pd
    <imports>
    # plan -
    def plot(data: pd.DataFrame):
        <stub> # only modify this section
        return plt;

    chart = plot(data) # data already contains the data to be plotted. Always include this line. No additional code beyond this line."""

    messages = [
        {"role": "system", "content": vis_system_prompt},
        {"role": "system", "content": f"The dataset summary is : {data_summary} \n\n"},
        library_instructions,
        {"role": "user",
            "content":
            f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n"}]


    # response = openai_client.chat.completions.create(
    #     # model="gpt-4",  
    #     model = "gpt-3.5-turbo",
    #     messages = messages
    # )

    # 提取生成的文本
    # generated_text_insight_code = response.choices[0].message.content
    generated_text_insight_code ="""python
    
    # plan -
    import matplotlib.pyplot as plt
    import pandas as pd
    def plot(data: pd.DataFrame):
        # Get the count of each car type
        type_counts = data['Type'].value_counts()

        # Plot the bar chart
        plt.bar(type_counts.index, type_counts.values)

        # Set the x-axis label
        plt.xlabel('Car Type')

        # Set the y-axis label
        plt.ylabel('Count')

        # Add a legend
        plt.legend(['Count'])

        return plt;

    chart = plot(data)"""

    print(generated_text_insight_code)
    return generated_text_insight_code

def get_chart_file_path(code,file_name):
    code = textwrap.dedent(code)
    code = f"""
# plan -
def plot(data: pd.DataFrame):
    import matplotlib.pyplot as plt
    import pandas as pd
    print("--------")
    # Get the count of each car type
    type_counts = data['Type'].value_counts()

    # Plot the bar chart
    plt.bar(type_counts.index, type_counts.values)

    # Set the x-axis label
    plt.xlabel('Car Type')

    # Set the y-axis label
    plt.ylabel('Count')
    print("--------")
    # Add a legend
    plt.legend(['Count'])
    
    return plt

data = pd.read_csv('./data/cars.csv')
chart = plot(data)
chart.savefig('./image/{file_name}.png')
"""
    exec(code)
    return f'./image/{file_name}.png'