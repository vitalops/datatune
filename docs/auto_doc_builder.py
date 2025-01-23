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

def format_docstring(content):
    """Format docstring to proper markdown."""
    lines = content.split('\n')
    formatted = []
    in_args = False
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # Handle headers
        if line.startswith('#'):
            in_args = False
            current_section = None
            formatted.append(f"\n{line}\n")
            continue
            
        # Handle section markers
        if line.lower().startswith(('args:', 'returns:', 'raises:')):
            section = line.split(':')[0]
            formatted.append(f"\n**{section}:**\n")
            in_args = section.lower() == 'args'
            current_section = section.lower()
            continue
            
        # Format argument descriptions
        if in_args and ':' in line:
            param, desc = line.split(':', 1)
            formatted.append(f"* **{param.strip()}** - {desc.strip()}")
        elif current_section and line:
            if line[0].isspace():  # Continuation of previous item
                formatted.append(f"  {line.strip()}")
            else:
                formatted.append(line)
                
    return '\n'.join(formatted)

def generate_toctree(docs_path):
    """Generate toctree content for documentation files."""
    toctree_content = ['```{toctree}', ':maxdepth: 2', ':caption: Contents:\n']
    
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.md'):
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
        if '```{toctree}' in existing_content:
            before_toctree = existing_content.split('```{toctree}')[0]
            after_toctree = existing_content.split('```\n')[-1]
            new_content = (
                f"{before_toctree.rstrip()}\n\n"
                f"{generate_toctree(docs_path)}\n"
                f"{after_toctree.lstrip()}"
            )
        else:
            new_content = (
                f"{existing_content.rstrip()}\n\n"
                f"{generate_toctree(docs_path)}\n"
            )
    else:
        new_content = (
            "# Documentation\n\n"
            f"{generate_toctree(docs_path)}\n"
        )
    
    with open(output_path / 'index.md', 'w') as f:
        f.write(new_content)

def copy_docs(docs_path, output_path):
    """Copy documentation files while preserving existing structure."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    for root, _, files in os.walk(docs_path):
        root_path = Path(root)
        
        for file in files:
            if file.endswith('.md'):
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
                
                with open(src_file, 'r') as src:
                    content = src.read()
                
                formatted_content = format_docstring(content)
                
                with open(dst_file, 'w') as dst:
                    dst.write(formatted_content)

def build_docs(docs_path=None, output_path=None):
    """Main function to build documentation."""
    docs_path = Path(docs_path or '.')
    output_path = Path(output_path or 'source')
    
    copy_docs(docs_path, output_path)
    generate_index(docs_path, output_path)

if __name__ == '__main__':
    build_docs()