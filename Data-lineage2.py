
import pandas as pd  # Make sure pandas is imported

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
            "input_datasets": ", ".join(recipe_info['inputs']),
            "output_dataset": final_ds_fullname
        })

def get_recipe_sequence(final_datasets, recipe_graph):
    recipe_sequence = []
    visited = set()
    for final_ds in final_datasets:
        trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)
    return recipe_sequence

# Now, let's call this for all final datasets in the lineage
all_recipe_sequences = {}

for project_key, zones in full_lineage.items():
    for zone, datasets in zones.items():
        for final_dataset, steps in datasets.items():
            # Rebuild recipe graph if needed (assuming it's already built from the lineage code)
            recipe_graph = {}
            for step in steps:
                recipe_graph[step["output_dataset"]] = {
                    "recipe_name": step["recipe_name"],
                    "recipe_type": step["recipe_type"],
                    "inputs": step["input_datasets"].split(", ")
                }

            # Get the recipe sequence for the final dataset
            recipe_sequence = get_recipe_sequence([final_dataset], recipe_graph)

            # Store the result in a dictionary
            all_recipe_sequences[f"{project_key}.{zone}.{final_dataset}"] = recipe_sequence

# Convert to DataFrame
rows = []
for final_dataset, sequence in all_recipe_sequences.items():
    for step in sequence:
        rows.append({
            "final_dataset": final_dataset,
            "recipe_name": step["recipe_name"],
            "recipe_type": step["recipe_type"],
            "input_datasets": step["input_datasets"],
            "output_dataset": step["output_dataset"]
        })

df = pd.DataFrame(rows)
