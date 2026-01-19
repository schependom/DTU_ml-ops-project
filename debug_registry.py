import wandb
import sys

entity = "schependom"
if len(sys.argv) > 1:
    entity = sys.argv[1]

print(f"Listing projects for entity: {entity}")
api = wandb.Api()
try:
    projects = api.projects(entity)
    found = False
    for p in projects:
        print(f"- Name: {p.name}, ID: {p.id}, URL: {p.url}")
        if "registry" in p.name.lower():
            print(f"  ^^^ POTENTIAL MATCH ^^^")
            found = True
    if not found:
        print("No projects found with 'registry' in the name.")
except Exception as e:
    print(f"Error: {e}")
