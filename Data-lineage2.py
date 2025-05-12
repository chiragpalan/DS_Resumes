
import dataiku                                  # 1
import pandas as pd                            # 2
from collections import defaultdict              # 3

client = dataiku.api_client()                    # 4
projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]  # 5

full_lineage = defaultdict(lambda: defaultdict(dict))  # 6

def get_zone_name(dataset_def):                 # 7
    return dataset_def.get("zone", "default")   # 8

def trace_recipe_chain(final_ds_fullname, recipe_graph, visited, recipe_sequence):  # 9
    if final_ds_fullname in visited:            # 10
        return                                  # 11

    visited.add(final_ds_fullname)              # 12
    recipe_info = recipe_graph.get(final_ds_fullname)  # 13
    
    if recipe_info:                             # 14
        for inp in recipe_info['inputs']:       # 15
            trace_recipe_chain(inp, recipe_graph, visited, recipe_sequence)  # 16
        recipe_sequence.append({                          # 17
            "recipe_name": recipe_info['recipe_name'],               # 18
            "recipe_type": recipe_info['recipe_type'],               # 19
            "input_datasets": ", ".join(recipe_info['inputs']),      # 20
            "output_dataset": final_ds_fullname                     # 21
        })                                       # 22

def get_recipe_sequence(final_datasets, recipe_graph):  # 23
    recipe_sequence = []                           # 24
    visited = set()                                 # 25
    for final_ds in final_datasets:                 # 26
        trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)  # 27
    return recipe_sequence                          # 28

# Main code to gather lineage and trace recipes
all_recipe_sequences = {}  # 29

for project_key in projects_to_include:         # 30
    print(f"Processing project: {project_key}") # 31
    project = client.get_project(project_key)   # 32

    datasets = project.list_datasets()           # 33
    dataset_names = {d['name'] for d in datasets}  # 34
    dataset_fullnames = {d['name']: f"{project_key}.{d['name']}" for d in datasets}  # 35

    recipes = project.list_recipes()            # 36
    recipe_graph = {}                           # 37
    all_input_datasets = set()                  # 38
    output_to_zone = {}                         # 39

    for rec in recipes:                         # 40
        recipe_name = rec["name"]               # 41
        recipe_obj = project.get_recipe(recipe_name)  # 42

        try:                                     # 43
            settings = recipe_obj.get_settings()  # 44
            raw_def = settings.get_recipe_raw_definition()  # 45
        except Exception as e:                   # 46
            print(f"Failed to get settings for recipe {recipe_name}: {e}")  # 47
            continue                             # 48

        recipe_type = raw_def.get("type", "unknown")  # 49

        inputs = []                              # 50
        outputs = []                             # 51

        try:                                     # 52
            for role, role_data in raw_def.get("inputs", {}).items():       # 53
                for inp in role_data.get("items", []):                      # 54
                    inp_ref = inp["ref"]                                    # 55
                    if "." not in inp_ref:                                  # 56
                        inp_ref = f"{project_key}.{inp_ref}"                # 57
                    inputs.append(inp_ref)                                  # 58

            for role, role_data in raw_def.get("outputs", {}).items():      # 59
                for out in role_data.get("items", []):                      # 60
                    out_ref = out["ref"]                                    # 61
                    if "." not in out_ref:                                  # 62
                        out_ref = f"{project_key}.{out_ref}"                # 63
                    outputs.append(out_ref)                                 # 64
        except Exception as e:                                              # 65
            print(f"Error parsing recipe {recipe_name}: {e}")              # 66
            continue                                                        # 67

        for output in outputs:                                              # 68
            recipe_graph[output] = {                                        # 69
                "recipe_name": recipe_name,                                 # 70
                "recipe_type": recipe_type,                                 # 71
                "inputs": inputs                                            # 72
            }                                                               # 73
            all_input_datasets.update(inputs)                               # 74

            try:                                                            # 75
                ds_obj = project.get_dataset(output.split('.')[-1])         # 76
                ds_def = ds_obj.get_definition()                            # 77
                zone = get_zone_name(ds_def)                                # 78
            except:                                                         # 79
                zone = "unknown"                                            # 80
            output_to_zone[output] = zone                                   # 81

    all_project_datasets = {f"{project_key}.{d['name']}" for d in datasets}  # 82
    final_datasets = all_project_datasets - all_input_datasets               # 83

    for final_ds in final_datasets:                                          # 84
        zone = output_to_zone.get(final_ds, "unknown")                       # 85
        visited = set()                                                      # 86
        steps = trace_recipe_chain(final_ds, recipe_graph, visited, [])      # 87
        full_lineage[project_key][zone][final_ds] = steps                   # 88

# Trace recipes for each final dataset
all_recipe_sequences = {}  # 89
for project_key, zones in full_lineage.items():                          # 90
    for zone, datasets in zones.items():                                 # 91
        for final_dataset, steps in datasets.items():                     # 92
            recipe_graph = {}                                              # 93
            for step in steps:                                             # 94
                recipe_graph[step["output_dataset"]] = {                   # 95
                    "recipe_name": step["recipe_name"],                    # 96
                    "recipe_type": step["recipe_type"],                    # 97
                    "inputs": step["input_datasets"].split(", ")           # 98
                }                                                          # 99

            # Get the sequence of recipes to run
            recipe_sequence = get_recipe_sequence([final_dataset], recipe_graph)  # 100
            all_recipe_sequences[f"{project_key}.{zone}.{final_dataset}"] = recipe_sequence  # 101

# Convert to DataFrame
rows = []  # 102
for final_dataset, sequence in all_recipe_sequences.items():           # 103
    for step in sequence:                                              # 104
        rows.append({                                                   # 105
            "final_dataset": final_dataset,                              # 106
            "recipe_name": step["recipe_name"],                          # 107
            "recipe_type": step["recipe_type"],                          # 108
            "input_datasets": step["input_datasets"],                     # 109
            "output_dataset": step["output_dataset"]                      # 110
        })                                                               # 111

df = pd.DataFrame(rows)                                                 # 112

# Export to Excel (optional)
df.to_excel("dataiku_lineage.xlsx", index=False)                        # 113
