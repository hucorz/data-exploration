import json

def get_recommand_field(*args, **kwargs):
    pass

def get_data_summary(*args, **kwargs):
    """
    Returns a summary of the dataset in the form of a dict
    """
    res = {
        "name": "cars.csv",
        "file_name": "cars.csv",
        "dataset_description": "",
        "fields": [
            {
                "column": "Name",
                "properties": {
                    "dtype": "string",
                    "num_unique_values": 383,
                    "semantic_type": "",
                    "description": "",
                    "code": """
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)""",
                },
            },
            {
                "column": "Type",
                "properties": {
                    "dtype": "category",
                    "num_unique_values": 5,
                    "semantic_type": "",
                    "description": "",
                    "code": """
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)""",
                },
            },
            {
                "column": "Type",
                "properties": {
                    "dtype": "category",
                    "num_unique_values": 5,
                    "semantic_type": "",
                    "description": "",
                    "code": """
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)""",
                },
            },
            {
                "column": "Type",
                "properties": {
                    "dtype": "category",
                    "num_unique_values": 5,
                    "semantic_type": "",
                    "description": "",
                    "code": """
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)""",
                },
            },
            {
                "column": "Type",
                "properties": {
                    "dtype": "category",
                    "num_unique_values": 5,
                    "semantic_type": "",
                    "description": "",
                    "code": """
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)""",
                },
            },
        ],
    }
    return res
