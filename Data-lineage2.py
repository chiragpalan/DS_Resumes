
1  import dataiku
2  import pandas as pd
3  from collections import defaultdict
4
5  client = dataiku.api_client()
6  projects_to_include = ["PROJECT1", "PROJECT2", "PROJECT3", "PROJECT4"]  # Update as needed
7
8  full_lineage = defaultdict(lambda: defaultdict(dict))
9
10 def get_zone_name(dataset_def):
11     return dataset_def.get("zone", "default")
12
13 def trace_recipe_chain(final_ds_fullname, recipe_graph, visited, recipe_sequence):
14     if final_ds_fullname in visited:
15         return
16     visited.add(final_ds_fullname)
17     recipe_info = recipe_graph.get(final_ds_fullname)
18     if recipe_info:
19         for inp in recipe_info['inputs']:
20             trace_recipe_chain(inp, recipe_graph, visited, recipe_sequence)
21         recipe_sequence.append({
22             "recipe_name": recipe_info['recipe_name'],
23             "recipe_type": recipe_info['recipe_type'],
24             "input_datasets": recipe_info['inputs'],
25             "output_dataset": final_ds_fullname
26         })
27
28 def get_recipe_sequence(final_datasets, recipe_graph):
29     recipe_sequence = []
30     visited = set()
31     for final_ds in final_datasets:
32         trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)
33     return recipe_sequence
34
35 all_recipe_sequences = {}
36
37 for project_key in projects_to_include:
38     print(f"Processing project: {project_key}")
39     project = client.get_project(project_key)
40
41     datasets = project.list_datasets()
42     dataset_names = {d['name'] for d in datasets}
43     dataset_fullnames = {d['name']: f"{project_key}.{d['name']}" for d in datasets}
44
45     recipes = project.list_recipes()
46     recipe_graph = {}
47     all_input_datasets = set()
48     output_to_zone = {}
49
50     for rec in recipes:
51         recipe_name = rec["name"]
52         recipe_obj = project.get_recipe(recipe_name)
53
54         try:
55             settings = recipe_obj.get_settings()
56             raw_def = settings.get_recipe_raw_definition()
57         except Exception as e:
58             print(f"Failed to get settings for recipe {recipe_name}: {e}")
59             continue
60
61         recipe_type = raw_def.get("type", "unknown")
62         inputs = []
63         outputs = []
64
65         try:
66             for role, role_data in raw_def.get("inputs", {}).items():
67                 for inp in role_data.get("items", []):
68                     inp_ref = inp["ref"]
69                     if "." not in inp_ref:
70                         inp_ref = f"{project_key}.{inp_ref}"
71                     inputs.append(inp_ref)
72             for role, role_data in raw_def.get("outputs", {}).items():
73                 for out in role_data.get("items", []):
74                     out_ref = out["ref"]
75                     if "." not in out_ref:
76                         out_ref = f"{project_key}.{out_ref}"
77                     outputs.append(out_ref)
78         except Exception as e:
79             print(f"Error parsing recipe {recipe_name}: {e}")
80             continue
81
82         for output in outputs:
83             recipe_graph[output] = {
84                 "recipe_name": recipe_name,
85                 "recipe_type": recipe_type,
86                 "inputs": inputs
87             }
88             all_input_datasets.update(inputs)
89
90             try:
91                 ds_obj = project.get_dataset(output.split('.')[-1])
92                 ds_def = ds_obj.get_definition()
93                 zone = get_zone_name(ds_def)
94             except:
95                 zone = "unknown"
96             output_to_zone[output] = zone
97
98     all_project_datasets = {f"{project_key}.{d['name']}" for d in datasets}
99     final_datasets = all_project_datasets - all_input_datasets
100
101     for final_ds in final_datasets:
102         zone = output_to_zone.get(final_ds, "unknown")
103         visited = set()
104         recipe_sequence = []
105         trace_recipe_chain(final_ds, recipe_graph, visited, recipe_sequence)
106         steps = recipe_sequence
107         full_lineage[project_key][zone][final_ds] = steps
108
109 all_recipe_sequences = {}
110 for project_key, zones in full_lineage.items():
111     for zone, datasets in zones.items():
112         for final_dataset, steps in datasets.items():
113             recipe_graph = {}
114             for step in steps:
115                 recipe_graph[step["output_dataset"]] = {
116                     "recipe_name": step["recipe_name"],
117                     "recipe_type": step["recipe_type"],
118                     "inputs": step["input_datasets"]
119                 }
120             recipe_sequence = get_recipe_sequence([final_dataset], recipe_graph)
121             all_recipe_sequences[f"{project_key}.{zone}.{final_dataset}"] = recipe_sequence
122
123 rows = []
124 for final_dataset, sequence in all_recipe_sequences.items():
125     for step in sequence:
126         rows.append({
127             "final_dataset": final_dataset,
128             "recipe_name": step["recipe_name"],
129             "recipe_type": step["recipe_type"],
130             "input_datasets": ", ".join(step["input_datasets"]),
131             "output_dataset": step["output_dataset"]
132         })
133
134 df = pd.DataFrame(rows)
135 df.to_excel("dataiku_lineage.xlsx", index=False)
