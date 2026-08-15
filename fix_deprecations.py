import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

original = content

# Replace use_container_width=True -> width="stretch"
content = content.replace("use_container_width=True", 'width="stretch"')

# Replace use_container_width=False -> width="content"
content = content.replace("use_container_width=False", 'width="content"')

count = original.count("use_container_width=True") + original.count("use_container_width=False")
print(f"Fixed {count} occurrences of use_container_width")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("app.py updated successfully.")
