#!/usr/bin/env python3
"""
Extract text from a corrupted tar archive.
The tar archive appears to be damaged, so we'll read it more carefully
and extract as much as we can.
"""
import sys

def extract_from_damaged_tar(tar_file, output_file):
    """
    Extract text from a damaged tar archive by manually parsing blocks.
    """
    documents = []
    
    with open(tar_file, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data):,} bytes ({len(data) / 1024 / 1024:.1f} MB)")
    
    # Tar archives have 512-byte headers followed by file content
    # Header format: filename (100 bytes) + metadata (412 bytes) + data
    pos = 0
    file_count = 0
    errors = 0
    
    while pos < len(data):
        # Check if we have enough data for a header
        if pos + 512 > len(data):
            break
        
        # Read tar header (512 bytes)
        header = data[pos:pos+512]
        
        # Check if this is a valid tar header (has "ustar" magic)
        if b'ustar' not in header:
            # Not a valid tar header, might be end of archive or corruption
            # Try skipping ahead to find next valid header
            pos += 512
            continue
        
        # Extract filename (first 100 bytes, null-terminated)
        name = header[:100].rstrip(b'\x00').decode('utf-8', errors='ignore')
        if not name:
            pos += 512
            continue
        
        # Extract file size (bytes 124-136, octal)
        try:
            size_str = header[124:136].decode('ascii').strip().rstrip('\x00')
            if not size_str:
                pos += 512
                continue
            file_size = int(size_str, 8)  # Octal number
        except:
            # Skip damaged header
            pos += 512
            continue
        
        # Calculate content position
        content_start = pos + 512
        content_end = content_start + file_size
        
        if content_end > len(data):
            break
        
        # Extract content
        content = data[content_start:content_end]
        
        try:
            # Decode text
            text = content.decode('utf-8', errors='ignore').strip()
            
            # Replace newlines with spaces
            text = text.replace('\r\n', ' ').replace('\n', ' ')
            text = ' '.join(text.split())
            
            if len(text) > 50:
                documents.append(text)
                file_count += 1
                if file_count % 10000 == 0:
                    print(f"Extracted {file_count:,} documents...", end='\r')
                
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"\nWarning: Error extracting file at position {pos}: {e}")
        
        # Move to next tar block (512-byte aligned)
        file_size_blocks = (file_size + 511) // 512
        pos += 512 + (file_size_blocks * 512)
    
    print(f"\nTotal files extracted: {file_count:,}")
    print(f"Errors encountered: {errors}")
    
    # Write documents to file
    if documents:
        print(f"Writing {len(documents):,} documents to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc + '\n')
    
    return len(documents)

def main():
    input_file = 'processed_data/train_subset_clean.txt'
    output_file = 'processed_data/train_subset_cleaned.txt'
    
    print(f"🔍 Extracting text from damaged tar archive: {input_file}")
    print("=" * 60)
    print("Note: The tar archive appears to be damaged, but we can still")
    print("extract the text content manually by reading through the file.")
    print("=" * 60)
    
    num_docs = extract_from_damaged_tar(input_file, output_file)
    
    if num_docs and num_docs > 100:
        print("=" * 60)
        print(f"✅ Done! Extracted {num_docs:,} documents")
        print(f"📄 Output saved to: {output_file}")
        print("\nNext steps:")
        print("1. Check the output file to verify it looks correct:")
        print(f"   head -5 {output_file}")
        print(f"   wc -l {output_file}")
        print("2. If it looks good, replace the original:")
        print(f"   mv {output_file} {input_file}")
        print("3. Commit and push to GitHub")
    else:
        print("❌ Failed or extracted too few documents")
        sys.exit(1)

if __name__ == '__main__':
    main()


