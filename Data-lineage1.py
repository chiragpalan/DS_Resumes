
import dataiku
from collections import defaultdict

client = dataiku.api_client()
projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]

# Structure: {project: {zone: {final_dataset: [steps]}}}
full_lineage = defaultdict(lambda: defaultdict(dict))

def get_zone_name(dataset_def):
    return dataset_def.get("zone", "default")

def trace_recipe_chain(final_ds_fullname, recipe_graph, visited):
    if final_ds_fullname in visited:
        return []

    visited.add(final_ds_fullname)
    recipe_info = recipe_graph.get(final_ds_fullname)
    steps = []

    if recipe_info:
        for inp in recipe_info['inputs']:
            steps += trace_recipe_chain(inp, recipe_graph, visited)
        steps.append({
            "recipe_name": recipe_info['recipe_name'],
            "recipe_type": recipe_info['recipe_type'],
            "output_dataset": final_ds_fullname,
            "input_datasets": recipe_info['inputs']
        })
    return steps

for project_key in projects_to_include:
    print(f"Processing project: {project_key}")
    project = client.get_project(project_key)

    datasets = project.list_datasets()
    dataset_names = {d['name'] for d in datasets}
    dataset_fullnames = {d['name']: f"{project_key}.{d['name']}" for d in datasets}

    recipes = project.list_recipes()
    recipe_graph = {}
    all_input_datasets = set()
    output_to_zone = {}

    for rec in recipes:
        recipe_name = rec["name"]
        recipe_obj = project.get_recipe(recipe_name)

        try:
            settings = recipe_obj.get_settings()
            raw_def = settings.get_recipe_raw_definition()
        except Exception as e:
            print(f"Failed to get settings for recipe {recipe_name}: {e}")
            continue

        recipe_type = raw_def.get("type", "unknown")

        inputs = []
        outputs = []

        try:
            for role, role_data in raw_def.get("inputs", {}).items():
                for inp in role_data.get("items", []):
                    inp_ref = inp["ref"]
                    if "." not in inp_ref:
                        inp_ref = f"{project_key}.{inp_ref}"
                    inputs.append(inp_ref)

            for role, role_data in raw_def.get("outputs", {}).items():
                for out in role_data.get("items", []):
                    out_ref = out["ref"]
                    if "." not in out_ref:
                        out_ref = f"{project_key}.{out_ref}"
                    outputs.append(out_ref)
        except Exception as e:
            print(f"Error parsing recipe {recipe_name}: {e}")
            continue

        for output in outputs:
            recipe_graph[output] = {
                "recipe_name": recipe_name,
                "recipe_type": recipe_type,
                "inputs": inputs
            }
            all_input_datasets.update(inputs)

            try:
                ds_obj = project.get_dataset(output.split('.')[-1])
                ds_def = ds_obj.get_definition()
                zone = get_zone_name(ds_def)
            except:
                zone = "unknown"
            output_to_zone[output] = zone

    # Determine final datasets
    all_project_datasets = {f"{project_key}.{d['name']}" for d in datasets}
    final_datasets = all_project_datasets - all_input_datasets

    for final_ds in final_datasets:
        zone = output_to_zone.get(final_ds, "unknown")
        visited = set()
        steps = trace_recipe_chain(final_ds, recipe_graph, visited)
        full_lineage[project_key][zone][final_ds] = steps

# Example to print or export
# import json
# print(json.dumps(full_lineage, indent=2))
