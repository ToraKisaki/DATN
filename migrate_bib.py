import os
import re

def normalize_title(title):
    t = re.sub(r'[\{\}]', '', title)
    t = t.lower()
    t = re.sub(r'[^a-z0-9]', '', t)
    return t

def parse_bib_robust(file_path):
    entries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    blocks = content.split('@')
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        if not lines: continue
        first_line = lines[0]
        # find the key
        match = re.search(r'^[a-zA-Z0-9_]+\s*\{\s*([^,]+),', first_line)
        if not match:
            # might be on next line or same line
            match = re.search(r'^[a-zA-Z0-9_]+\s*\{\s*([^,]+),', block)
            if not match:
                continue
        key = match.group(1).strip()
        
        # Extract title
        # Find 'title'
        title_idx = block.lower().find('title')
        if title_idx == -1:
            continue
        
        # find the first { or " after title_idx
        eq_idx = block.find('=', title_idx)
        if eq_idx == -1: continue
        
        start_char_idx = -1
        for i in range(eq_idx+1, len(block)):
            if block[i] in ['{', '"']:
                start_char_idx = i
                break
        
        if start_char_idx == -1: continue
        
        open_char = block[start_char_idx]
        close_char = '}' if open_char == '{' else '"'
        
        title_content = ""
        if open_char == '{':
            bracket_count = 1
            for i in range(start_char_idx+1, len(block)):
                if block[i] == '{':
                    bracket_count += 1
                elif block[i] == '}':
                    bracket_count -= 1
                
                if bracket_count == 0:
                    break
                title_content += block[i]
        else:
            for i in range(start_char_idx+1, len(block)):
                if block[i] == '"':
                    break
                title_content += block[i]
                
        norm_title = normalize_title(title_content)
        if norm_title:
            entries[key] = norm_title
            
    return entries

def main():
    old_bib = 'datn-git.bib'
    new_bib = 'datn.bib'
    
    old_entries = parse_bib_robust(old_bib)
    new_entries = parse_bib_robust(new_bib)
    
    new_by_title = {v: k for k, v in new_entries.items()}
    
    mapping = {}
    for old_k, norm_title in old_entries.items():
        if norm_title in new_by_title:
            mapping[old_k] = new_by_title[norm_title]
        else:
            # Let's try to find a substring match
            found = False
            for new_title, new_k in new_by_title.items():
                if norm_title in new_title or new_title in norm_title:
                    mapping[old_k] = new_k
                    found = True
                    break
            if not found:
                print(f"Could not find match for old key: {old_k} (Title: {norm_title})")
            
    print(f"Found {len(mapping)} mappings out of {len(old_entries)} old entries.")
    
    tex_files = []
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'venv' in root:
            continue
        for f in files:
            if f.endswith('.tex'):
                tex_files.append(os.path.join(root, f))
                
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Skipping {tex_file} due to encoding issue.")
            continue
            
        new_content = content
        replaced = False
        for old_k, new_k in mapping.items():
            pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(old_k) + r'(?![a-zA-Z0-9_-])'
            new_content_after = re.sub(pattern, new_k, new_content)
            if new_content_after != new_content:
                replaced = True
            new_content = new_content_after
            
        if replaced:
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {tex_file}")
    
    # Also we should change the \bibliography{datn-git} to \bibliography{datn} in DoAn.tex or similar files.
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        new_content = re.sub(r'\\addbibresource\{datn-git\.bib\}', r'\\addbibresource{datn.bib}', content)
        new_content = re.sub(r'\\bibliography\{datn-git\}', r'\\bibliography{datn}', new_content)
        
        if new_content != content:
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated bibliography reference in {tex_file}")

if __name__ == '__main__':
    main()
