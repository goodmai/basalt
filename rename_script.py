import os
import re

# Replacements for contents
replacements = {
    'GemmaServerCore': 'GemCore',
    'GemmaServerError': 'GemError',
    'GemmaServerBin': 'GemBin',
    'GemmaServerTests': 'GemTests',
    'GemmaServer': 'Gem',
    'gemmaserver': 'gem'
}

# Directories and files to rename
paths_to_rename = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.build' in root:
        continue
    for d in dirs:
        if 'GemmaServer' in d:
            paths_to_rename.append((os.path.join(root, d), os.path.join(root, d.replace('GemmaServer', 'Gem'))))
    for f in files:
        if 'GemmaServer' in f:
            paths_to_rename.append((os.path.join(root, f), os.path.join(root, f.replace('GemmaServer', 'Gem'))))

# Sort paths by depth descending so we rename deepest first
paths_to_rename.sort(key=lambda x: x[0].count(os.sep), reverse=True)

for old_path, new_path in paths_to_rename:
    print(f"Renaming {old_path} to {new_path}")
    os.rename(old_path, new_path)

# Update file contents
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.build' in root or '.archive' in root:
        continue
    for f in files:
        if not (f.endswith('.swift') or f.endswith('.md') or f.endswith('.sh') or f.endswith('.rb') or f == 'Package.swift' or f == 'Package.resolved'):
            continue
        filepath = os.path.join(root, f)
        try:
            with open(filepath, 'r') as file:
                content = file.read()
            
            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace(old, new)
            
            if content != new_content:
                print(f"Updating contents of {filepath}")
                with open(filepath, 'w') as file:
                    file.write(new_content)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

