
import dataiku
from collections import defaultdict

client = dataiku.api_client()
projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]

full_lineage = defaultdict(lambda: defaultdict(dict))

def get_zone_name(dataset_def):
    return dataset_def.get("zone", "default")

def trace_recipe_chain(project, final_dataset, recipe_graph):
    visited = set()
    steps = []

    def dfs(ds_name):
        if ds_name in visited:
            return
        visited.add(ds_name)

        recipe_info = recipe_graph.get(ds_name)
        if recipe_info:
            for inp in recipe_info['inputs']:
                dfs(inp)
            steps.append({
                "step": len(steps) + 1,
                "recipe_name": recipe_info['recipe_name'],
                "recipe_type": recipe_info['recipe_type'],
                "output_dataset": ds_name,
                "input_datasets": recipe_info['inputs']
            })

    dfs(final_dataset)
    return steps

for project_key in projects_to_include:
    print(f"Processing project: {project_key}")
    project = client.get_project(project_key)
    recipes = project.list_recipes()
    datasets = project.list_datasets()

    recipe_graph = {}
    all_inputs = set()
    output_to_zone = {}

    for rec in recipes:
        recipe_name = rec['name']
        recipe_obj = project.get_recipe(recipe_name)

        try:
            settings_obj = recipe_obj.get_settings()
            settings = settings_obj.recipe if hasattr(settings_obj, "recipe") else settings_obj.to_json()
        except Exception as e:
            print(f"Failed to get settings for recipe {recipe_name}: {e}")
            continue

        try:
            recipe_type = settings["type"]
        except (KeyError, TypeError):
            recipe_type = "unknown"

        # Safe access to inputs and outputs
        inputs = []
        outputs = []

        try:
            if "inputs" in settings and "main" in settings["inputs"]:
                inputs = [inp["ref"] for inp in settings["inputs"]["main"]]
            if "outputs" in settings and "main" in settings["outputs"]:
                outputs = [out["ref"] for out in settings["outputs"]["main"]]
        except Exception as e:
            print(f"Error accessing inputs/outputs for {recipe_name}: {e}")
            continue

        for out in outputs:
            recipe_graph[out] = {
                "recipe_name": recipe_name,
                "recipe_type": recipe_type,
                "inputs": inputs
            }
            all_inputs.update(inputs)

            try:
                dataset_name = out.split('.')[-1]
                dataset_obj = project.get_dataset(dataset_name)
                dataset_def = dataset_obj.get_definition()
                zone_name = get_zone_name(dataset_def)
                output_to_zone[out] = zone_name
            except:
                output_to_zone[out] = "unknown"

    all_dataset_names = [f"{project_key}.{d['name']}" for d in datasets]
    final_datasets = [ds for ds in all_dataset_names if ds not in all_inputs]

    for final_ds in final_datasets:
        zone = output_to_zone.get(final_ds, "unknown")
        steps = trace_recipe_chain(project, final_ds, recipe_graph)
        full_lineage[project_key][zone][final_ds] = steps

# Uncomment below to see the output
# import json
# print(json.dumps(full_lineage, indent=2))
