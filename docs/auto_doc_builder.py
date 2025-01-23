import os
from pathlib import Path

def generate_index(docs_path, output_path):
    # Create the index.md file
    with open(output_path / 'index.md', 'w') as f:
        f.write('# Documentation\n\n')
        f.write('```{toctree}\n')
        f.write(':maxdepth: 2\n')
        f.write(':caption: Contents:\n\n')
        
        # Recursively find all markdown files
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith('.md'):
                    # Skip files in source directory to avoid duplicates
                    if 'source' in Path(root).parts:
                        continue
                    rel_path = os.path.relpath(
                        os.path.join(root, file), 
                        str(docs_path)
                    )
                    f.write(f'{rel_path}\n')
        
        f.write('```\n')

def copy_docs(docs_path, output_path):
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all markdown files
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.md'):
                # Skip files in source directory to avoid duplicates
                if 'source' in Path(root).parts:
                    continue
                    
                src_file = Path(root) / file
                rel_path = os.path.relpath(root, str(docs_path))
                dst_dir = output_path / rel_path
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / file
                
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

if __name__ == '__main__':
    docs_path = Path('.')  # Current directory
    output_path = Path('source')  # Sphinx source directory
    
    copy_docs(docs_path, output_path)
    generate_index(docs_path, output_path)