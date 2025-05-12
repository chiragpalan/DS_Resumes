
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

client = dataiku.api_client()
all_projects = client.list_project_keys()

lineage_rows = []

# Helper function to walk lineage recursively
def walk_lineage(dataset, visited, path):
    if dataset in visited:
        return
    visited.add(dataset)
    project_key, dataset_name = dataset.split('.')
    project = client.get_project(project_key)
    
    try:
        ds = project.get_dataset(dataset_name)
        definition = ds.get_definition()
        last_build_info = ds.get_last_build_info()
        zone_name = definition.get("zone", "default")
    except:
        return
    
    try:
        # Get the recipe that outputs this dataset
        recipe = ds.get_settings().get_raw().get("creationDetails", {}).get("recipe")
        if recipe:
            recipe_obj = project.get_recipe(recipe)
            recipe_type = recipe_obj.type
            input_datasets = [inp["ref"] for inp in recipe_obj.get_definition().get("inputs", {}).get("main", [])]
            
            # Add a row for current dataset
            lineage_rows.append({
                "Final Dataset": path[0] if path else dataset,
                "Intermediate Dataset": dataset,
                "Recipe Name": recipe,
                "Recipe Type": recipe_type,
                "Input Datasets": ', '.join(input_datasets),
                "Project Name": project_key,
                "Zone Name": zone_name,
                "Comments": ""
            })

            # Recurse for inputs
            for input_ds in input_datasets:
                walk_lineage(input_ds, visited, path + [input_ds])
    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")
        return

# Start lineage trace from all datasets across all projects
for project_key in all_projects:
    project = client.get_project(project_key)
    for ds_name in project.list_datasets():
        full_ds = f"{project_key}.{ds_name['name']}"
        walk_lineage(full_ds, set(), [full_ds])

# Create DataFrame and export to Excel
df = pd.DataFrame(lineage_rows)
df.to_excel("/tmp/dataset_lineage.xlsx", index=False)
