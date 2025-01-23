import os
from pathlib import Path
import shutil

def read_existing_index(output_path):
    """Read existing index.md if it exists in the source directory."""
    index_path = output_path / 'index.md'
    if index_path.exists():
        with open(index_path, 'r') as f:
            return f.read()
    return None

def generate_toctree(docs_path):
    """Generate toctree content for documentation files."""
    toctree_content = ['```{toctree}', ':maxdepth: 2', ':caption: Contents:\n']
    
    # Recursively find all markdown files
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.md'):
                # Skip files in source directory and index.md
                if 'source' in Path(root).parts or file == 'index.md':
                    continue
                rel_path = os.path.relpath(
                    os.path.join(root, file), 
                    str(docs_path)
                )
                toctree_content.append(rel_path)
    
    toctree_content.append('```')
    return '\n'.join(toctree_content)

def generate_index(docs_path, output_path):
    """Generate or update index.md while preserving existing content."""
    existing_content = read_existing_index(output_path)
    
    if existing_content:
        # Check if existing content already has a toctree
        if '```{toctree}' in existing_content:
            # Find and replace existing toctree
            before_toctree = existing_content.split('```{toctree}')[0]
            after_toctree = existing_content.split('```\n')[-1]
            new_content = (
                f"{before_toctree.rstrip()}\n\n"
                f"{generate_toctree(docs_path)}\n"
                f"{after_toctree.lstrip()}"
            )
        else:
            # Append toctree to existing content
            new_content = (
                f"{existing_content.rstrip()}\n\n"
                f"{generate_toctree(docs_path)}\n"
            )
    else:
        # Create new index.md with default content
        new_content = (
            "# Documentation\n\n"
            f"{generate_toctree(docs_path)}\n"
        )
    
    # Write the final content
    with open(output_path / 'index.md', 'w') as f:
        f.write(new_content)

def copy_docs(docs_path, output_path):
    """Copy documentation files while preserving existing structure."""
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all markdown files
    for root, _, files in os.walk(docs_path):
        root_path = Path(root)
        
        for file in files:
            if file.endswith('.md'):
                # Skip files in source directory and index.md in root
                if 'source' in root_path.parts or (
                    file == 'index.md' and 
                    root_path == docs_path
                ):
                    continue
                    
                src_file = root_path / file
                rel_path = os.path.relpath(root, str(docs_path))
                dst_dir = output_path / rel_path
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / file
                
                # Copy file if it doesn't exist or has been modified
                if not dst_file.exists() or (
                    os.path.getmtime(src_file) > os.path.getmtime(dst_file)
                ):
                    shutil.copy2(src_file, dst_file)

def build_docs(docs_path=None, output_path=None):
    """Main function to build documentation."""
    docs_path = Path(docs_path or '.')  # Current directory
    output_path = Path(output_path or 'source')  # Sphinx source directory
    
    copy_docs(docs_path, output_path)
    generate_index(docs_path, output_path)

if __name__ == '__main__':
    build_docs()