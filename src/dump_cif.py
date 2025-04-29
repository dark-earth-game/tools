import sys
import os
import numpy as np
from PIL import Image
import struct
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Reusing the functions from view_cif.py
def decompress_lzw(compressed_data, width, height):
    """
    LZW decompression for CIF files
    Note: This is a simplified implementation and may need adjustments based on the specific
    LZW variant used in the game.
    """
    # Basic LZW decompression
    code_size = 9
    max_code = 511  # 2^code_size - 1
    clear_code = 256
    end_code = 257
    next_code = 258
    
    dictionary = {i: bytes([i]) for i in range(256)}
    dictionary[clear_code] = clear_code
    dictionary[end_code] = end_code
    
    bits_buffer = 0
    bits_remaining = 0
    output = bytearray()
    old_code = -1
    pos = 0
    
    # Read bits from the compressed data
    def read_bits(n):
        nonlocal bits_buffer, bits_remaining, pos
        
        while bits_remaining < n and pos < len(compressed_data):
            bits_buffer |= compressed_data[pos] << bits_remaining
            bits_remaining += 8
            pos += 1
        
        value = bits_buffer & ((1 << n) - 1)
        bits_buffer >>= n
        bits_remaining -= n
        return value
    
    # Initial clear code
    code = read_bits(code_size)
    if code != clear_code:
        return None  # Should start with clear code
    
    code = read_bits(code_size)
    output.extend(dictionary[code])
    old_code = code
    
    while pos < len(compressed_data) and len(output) < width * height:
        code = read_bits(code_size)
        
        if code == end_code:
            break
        
        if code == clear_code:
            code_size = 9
            max_code = 511
            next_code = 258
            dictionary = {i: bytes([i]) for i in range(256)}
            dictionary[clear_code] = clear_code
            dictionary[end_code] = end_code
            
            code = read_bits(code_size)
            if code == end_code:
                break
                
            output.extend(dictionary[code])
            old_code = code
            continue
        
        if code < next_code:  # Code already in the dictionary
            entry = dictionary[code]
            output.extend(entry)
            if old_code >= 0:
                new_entry = dictionary[old_code] + bytes([entry[0]])
                dictionary[next_code] = new_entry
                next_code += 1
        else:  # Code not in the dictionary
            entry = dictionary[old_code]
            entry = entry + bytes([entry[0]])
            output.extend(entry)
            dictionary[next_code] = entry
            next_code += 1
        
        old_code = code
        
        # Increase code size when we've used all codes at current size
        if next_code > max_code and code_size < 12:
            code_size += 1
            max_code = (1 << code_size) - 1
    
    return bytes(output[:width*height])

def adjust_palette(palette, gamma=1.0, contrast=1.0, brightness=0):
    """
    Adjust palette with gamma, contrast and brightness
    
    Parameters:
    - palette: numpy array of shape (256, 3) containing RGB values
    - gamma: gamma correction factor (default 1.0, no change)
    - contrast: contrast adjustment factor (default 1.0, no change)
    - brightness: brightness adjustment (-255 to 255, default 0, no change)
    
    Returns:
    - adjusted palette
    """
    # Make a copy to avoid modifying the original
    adjusted = palette.astype(np.float32)
    
    # Apply gamma correction
    if gamma != 1.0:
        adjusted = 255.0 * ((adjusted / 255.0) ** (1.0 / gamma))
    
    # Apply contrast adjustment
    if contrast != 1.0:
        factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
        adjusted = factor * (adjusted - 128.0) + 128.0
    
    # Apply brightness adjustment
    if brightness != 0:
        adjusted += brightness
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

def convert_cif_to_png(cif_path, output_dir=None, overwrite=False, gamma=1.0, contrast=1.0, brightness=0):
    """Convert a CIF file to a PNG file"""
    try:
        # Determine output path
        if output_dir:
            # Create relative path within output directory
            rel_path = os.path.relpath(cif_path, start=args.input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.png')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            # Use same directory as CIF
            output_path = os.path.splitext(cif_path)[0] + '.png'
            
        # Skip if file exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {cif_path} (output already exists)")
            return False
        
        # Parse CIF file
        with open(cif_path, 'rb') as f:
            # Read header (16 bytes)
            header = f.read(16)
            version = struct.unpack('<I', header[0:4])[0]
            signature = header[4:8]
            width = struct.unpack('<I', header[8:12])[0]
            height = struct.unpack('<I', header[12:16])[0]

            print(f'Header: version={version}, signature={signature}, width={width}, height={height}')

            # Read color palette (768 bytes = 256 colors * 3 bytes)
            palette_data = f.read(0x300)
            
            # Process palette - right shift each byte by 2 as in the assembly code
            palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(256, 3)
            # palette = (palette >> 2).astype(np.uint8)
            
            # Apply gamma, contrast and brightness adjustments
            # only if file prefix starts with F
            if os.path.basename(cif_path).startswith('F'):
                palette = adjust_palette(palette, gamma, contrast, brightness)
            
            # Read the compressed image data
            compressed_data = f.read()
        
        # Decompress the LZW data
        image_data = decompress_lzw(compressed_data, width, height)
        
        if image_data is None or len(image_data) != width * height:
            print(f"Warning: Decompression incomplete for {cif_path}. Expected {width*height} bytes, got {len(image_data) if image_data else 0}")
            if image_data is None:
                image_data = bytes([0] * (width * height))
            else:
                # Pad with zeros if needed
                image_data = image_data + bytes([0] * (width * height - len(image_data)))
        
        # Create an indexed color image
        img_array = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width)
        
        # Convert to RGB using the palette
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                color_index = img_array[y, x]
                rgb_image[y, x] = palette[color_index]
        
        # Save as PNG
        pil_img = Image.fromarray(rgb_image)
        # turn red colour transparent
        pil_img = pil_img.convert("RGBA")
        data = pil_img.getdata()
        new_data = []
        for item in data:
            # change all white (also shades of whites)
            # pixels to transparent
            if item[0] in list(range(200, 256)) and item[1] in list(range(200, 256)) and item[2] in list(range(200, 256)):
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        pil_img.putdata(new_data)
        pil_img.save(output_path)
        
        # print(f"Converted: {cif_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        return False

def find_and_convert_cif_files(input_dir, output_dir=None, overwrite=False, max_workers=None, 
                             gamma=1.0, contrast=1.0, brightness=0):
    """Find all CIF files in the input directory recursively and convert them to PNG"""
    
    # Find all CIF files
    cif_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.cif'):
                cif_files.append(os.path.join(root, file))
    
    if not cif_files:
        print(f"No CIF files found in {input_dir}")
        return
    
    print(f"Found {len(cif_files)} CIF files")
    print(f"Image adjustments: gamma={gamma}, contrast={contrast}, brightness={brightness}")
    
    # Process files with ThreadPoolExecutor for parallel processing
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        future_to_file = {
            executor.submit(
                convert_cif_to_png, 
                file, 
                output_dir, 
                overwrite, 
                gamma, 
                contrast, 
                brightness
            ): file for file in cif_files
        }
        
        # Process results as they complete
        for future in future_to_file:
            if future.result():
                success_count += 1
    
    print(f"Conversion complete: {success_count} of {len(cif_files)} files converted successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CIF files to PNG recursively')
    parser.add_argument('input_dir', help='Input directory containing CIF files')
    parser.add_argument('-o', '--output-dir', help='Output directory for PNG files (keeps directory structure)')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing PNG files')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel conversion jobs')
    
    # Image adjustment options
    parser.add_argument('-g', '--gamma', type=float, default=1.0, 
                        help='Gamma correction (default: 1.0, values > 1 brighten midtones)')
    parser.add_argument('-c', '--contrast', type=float, default=1.0, 
                        help='Contrast adjustment (default: 1.0, values > 1 increase contrast)')
    parser.add_argument('-b', '--brightness', type=int, default=0,
                        help='Brightness adjustment (default: 0, range -255 to 255)')
    
    args = parser.parse_args()
    
    # Normalize paths
    args.input_dir = os.path.abspath(args.input_dir)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate adjustment parameters
    if args.gamma <= 0:
        print("Error: Gamma must be positive")
        sys.exit(1)
    if args.brightness < -255 or args.brightness > 255:
        print("Error: Brightness must be between -255 and 255")
        sys.exit(1)
    
    find_and_convert_cif_files(
        args.input_dir, 
        args.output_dir, 
        args.force, 
        args.jobs,
        args.gamma,
        args.contrast,
        args.brightness
    )
