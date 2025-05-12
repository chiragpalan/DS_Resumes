import dataiku
import pandas as pd
from collections import defaultdict

client = dataiku.api_client()
projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]  # Update with your project keys
dataiku_host = "https://<YOUR_DATAIKU_HOST>"  # Replace with your actual Dataiku URL

full_lineage = defaultdict(lambda: defaultdict(dict))

def get_zone_name(dataset_def):
    return dataset_def.get("zone", "default")

def get_flow_name_for_recipe(project, recipe_name):
    # Assuming the flow name is the same as the recipe name or you can extract it from settings
    flow_name = None
    try:
        recipe = project.get_recipe(recipe_name)
        flow_name = recipe.get_flow().get("name")  # Try to get flow name if it's available
    except Exception as e:
        print(f"Could not retrieve flow for recipe {recipe_name}: {e}")
    return flow_name or "default_flow"

def trace_recipe_chain(final_ds_fullname, recipe_graph, visited, recipe_sequence):
    if final_ds_fullname in visited:
        return
    visited.add(final_ds_fullname)
    recipe_info = recipe_graph.get(final_ds_fullname)
    if recipe_info:
        for inp in recipe_info['inputs']:
            trace_recipe_chain(inp, recipe_graph, visited, recipe_sequence)
        recipe_sequence.append({
            "recipe_name": recipe_info['recipe_name'],
            "recipe_type": recipe_info['recipe_type'],
            "input_datasets": recipe_info['inputs'],
            "output_dataset": final_ds_fullname
        })

def get_recipe_sequence(final_datasets, recipe_graph):
    recipe_sequence = []
    visited = set()
    for final_ds in final_datasets:
        trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)
    return recipe_sequence

all_recipe_sequences = {}

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

    all_project_datasets = {f"{project_key}.{d['name']}" for d in datasets}
    final_datasets = all_project_datasets - all_input_datasets

    for final_ds in final_datasets:
        zone = output_to_zone.get(final_ds, "unknown")
        visited = set()
        recipe_sequence = []
        trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)

        # Add recipe URLs with a trailing slash
        for step in recipe_sequence:
            # Flow URL with trailing slash
            flow_name = get_flow_name_for_recipe(project, step["recipe_name"])
            step["recipe_url"] = f"{dataiku_host}/projects/{project_key}/recipes/{step['recipe_name']}/"
            step["flow_url"] = f"{dataiku_host}/projects/{project_key}/flow/#{flow_name}/"

        full_lineage[project_key][zone][final_ds] = recipe_sequence

all_recipe_sequences = {}
for project_key, zones in full_lineage.items():
    for zone, datasets in zones.items():
        for final_dataset, steps in datasets.items():
            recipe_graph = {}
            for step in steps:
                recipe_graph[step["output_dataset"]] = {
                    "recipe_name": step["recipe_name"],
                    "recipe_type": step["recipe_type"],
                    "inputs": step["input_datasets"]
                }
            recipe_sequence = get_recipe_sequence([final_dataset], recipe_graph)

            # Add recipe URLs with trailing slash and flow URL
            for step in recipe_sequence:
                flow_name = get_flow_name_for_recipe(project, step["recipe_name"])
                step["recipe_url"] = f"{dataiku_host}/projects/{project_key}/recipes/{step['recipe_name']}/"
                step["flow_url"] = f"{dataiku_host}/projects/{project_key}/flow/#{flow_name}/"

            all_recipe_sequences[f"{project_key}.{zone}.{final_dataset}"] = recipe_sequence

# Create Excel
rows = []
for final_dataset, sequence in all_recipe_sequences.items():
    for step in sequence:
        rows.append({
            "final_dataset": final_dataset,
            "recipe_name": step["recipe_name"],
            "recipe_type": step["recipe_type"],
            "input_datasets": ", ".join(step["input_datasets"]),
            "output_dataset": step["output_dataset"],
            "recipe_url": step["recipe_url"],
            "flow_url": step["flow_url"]
        })

df = pd.DataFrame(rows)
df.to_excel("dataiku_lineage_with_links_and_flows.xlsx", index=False)
