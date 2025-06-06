import dataiku

# Define the zones you want to check (case-sensitive)
zones_to_check = ["Input Zone", "Processing"]

# Setup
client = dataiku.api_client()
project = client.get_default_project()
project_key = project.project_key
flow = project.get_flow()

# Get all flow zones
all_zones = flow.list_zones()

# Filter to only selected zones by name
selected_zones = [zone for zone in all_zones if zone.name in zones_to_check]

# Result lists
readable_datasets = []
unreadable_datasets = []

# Loop through datasets in selected zones only
for zone in selected_zones:
    zone_name = zone.name
    items = zone.items

    # Filter only datasets
    dataset_items = [item for item in items if item["type"] == "DATASET"]

    for item in dataset_items:
        dataset_name = item["ref"]
        try:
            dataset = dataiku.Dataset(dataset_name)
            df = dataset.get_dataframe(limit=10)
            readable_datasets.append((dataset_name, project_key, zone_name))
        except Exception as e:
            unreadable_datasets.append((dataset_name, project_key, zone_name))
            print(f"❌ Error reading dataset '{dataset_name}' in zone '{zone_name}': {e}")

# Output
print("\n✅ Readable Datasets in selected zones:")
for name, proj, zone in readable_datasets:
    print(f"  - {name} | Project: {proj} | Zone: {zone}")

print("\n🚫 Unreadable Datasets in selected zones:")
for name, proj, zone in unreadable_datasets:
    print(f"  - {name} | Project: {proj} | Zone: {zone}")
