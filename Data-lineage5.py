
import dataiku
from dataiku import api_client

client = api_client()
project = client.get_project("YOUR_PROJECT_KEY")
flow = project.get_flow()

# Get all datasets and recipes
datasets = flow.get_datasets()
recipes = flow.get_recipes()

# Map datasets to their inputs/outputs
recipe_info = []
dataset_outputs = set()

for recipe in recipes:
    details = recipe.get_definition()
    inputs = details['inputs']['main']
    outputs = details['outputs']['main']
    dataset_outputs.update(outputs)
    
    for output in outputs:
        recipe_info.append({
            "project": project.project_key,
            "output_dataset": output,
            "inputs": inputs,
            "recipe_name": recipe.name,
            "recipe_type": recipe.type
        })

# Find extreme-left datasets (no outputs to others)
all_dataset_names = set(d['name'] for d in datasets)
extreme_left_datasets = all_dataset_names - dataset_outputs

# Map zone if you use Flow Zones (optional)
dataset_zones = {}
for d in datasets:
    try:
        zone = d.get_zone()
        dataset_zones[d.name] = zone['name']
    except:
        dataset_zones[d.name] = 'Default'

# Format into final table
import pandas as pd

rows = []
for info in recipe_info:
    zone = dataset_zones.get(info['output_dataset'], 'Default')
    rows.append({
        "Project": info["project"],
        "Zone": zone,
        "Output Dataset": info["output_dataset"],
        "Recipe": info["recipe_name"],
        "Recipe Type": info["recipe_type"],
        "Input Datasets": ", ".join(i['ref'] for i in info["inputs"])
    })

df = pd.DataFrame(rows)

# Optionally filter only those ending at extreme-left datasets
if extreme_left_datasets:
    df = df[df["Output Dataset"].isin(extreme_left_datasets)]

df.head()
