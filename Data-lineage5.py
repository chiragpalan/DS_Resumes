
import dataiku
from dataiku import api_client
import pandas as pd

# Connect to the Dataiku API and retrieve the project
client = api_client()
project = client.get_project("YOUR_PROJECT_KEY")  # Replace with your actual project key

# List all datasets and recipes in the project
datasets = project.list_datasets()
recipes = project.list_recipes()

# Initialize sets and lists to store dataset and recipe information
all_dataset_names = set(d['name'] for d in datasets)
output_datasets = set()
recipe_info = []

# Iterate through each recipe to extract inputs and outputs
for recipe_meta in recipes:
    recipe = project.get_recipe(recipe_meta['name'])
    settings = recipe.get_settings()

    # Retrieve inputs and outputs from the recipe settings
    inputs = settings.get_recipe_inputs()
    outputs = settings.get_recipe_outputs()

    # Extract input and output dataset names
    input_dataset_names = []
    for role_inputs in inputs.values():
        input_dataset_names.extend(role_inputs)

    output_dataset_names = []
    for role_outputs in outputs.values():
        output_dataset_names.extend(role_outputs)

    output_datasets.update(output_dataset_names)

    for output in output_dataset_names:
        recipe_info.append({
            "Project": project.project_key,
            "Output Dataset": output,
            "Input Datasets": ", ".join(input_dataset_names),
            "Recipe": recipe_meta['name'],
            "Recipe Type": recipe_meta['type']
        })

# Identify datasets that are not used as inputs in any recipe (i.e., terminal datasets)
terminal_datasets = all_dataset_names - output_datasets

# Create a DataFrame from the recipe information
df = pd.DataFrame(recipe_info)

# Filter the DataFrame to include only terminal datasets
df = df[df["Output Dataset"].isin(terminal_datasets)]

# Display the resulting DataFrame
print(df.head())
