
import dataiku
from collections import defaultdict

client = dataiku.api_client()

# List of project keys to include
projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]

# Final output dictionary: project > zone > final_dataset > list of steps
full_lineage = defaultdict(lambda: defaultdict(dict))

def get_zone_name(dataset_def):
    # Extracts zone name from dataset definition (if available)
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

# Process each project
for project_key in projects_to_include:
    print(f"Processing project: {project_key}")
    project = client.get_project(project_key)
    recipes = project.list_recipes()
    datasets = project.list_datasets()

    # Build recipe graph and zone mapping
    recipe_graph = {}
    all_inputs = set()
    output_to_zone = {}

    for rec in recipes:
        recipe_name = rec['name']
        recipe_obj = project.get_recipe(recipe_name)
        definition = recipe_obj.get_definition()
        recipe_type = definition.get("type", "unknown")

        inputs = [inp['ref'] for inp in definition.get("inputs", {}).get("main", [])]
        outputs = [out['ref'] for out in definition.get("outputs", {}).get("main", [])]

        for out in outputs:
            recipe_graph[out] = {
                "recipe_name": recipe_name,
                "recipe_type": recipe_type,
                "inputs": inputs
            }
            all_inputs.update(inputs)

            # Determine zone (if possible)
            try:
                dataset_name = out.split('.')[-1]
                dataset_obj = project.get_dataset(dataset_name)
                dataset_def = dataset_obj.get_definition()
                zone_name = get_zone_name(dataset_def)
                output_to_zone[out] = zone_name
            except:
                output_to_zone[out] = "unknown"

    # Identify final datasets: those that are not inputs to any recipe
    all_dataset_names = [f"{project_key}.{d['name']}" for d in datasets]
    final_datasets = [ds for ds in all_dataset_names if ds not in all_inputs]

    # Trace lineage for each final dataset
    for final_ds in final_datasets:
        zone = output_to_zone.get(final_ds, "unknown")
        steps = trace_recipe_chain(project, final_ds, recipe_graph)
        full_lineage[project_key][zone][final_ds] = steps

# full_lineage now contains the full recipe lineage dictionary
# Example access:
# full_lineage["PROJECT1"]["zoneA"]["PROJECT1.final_dataset_x"]
