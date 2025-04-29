import os
import sys
import io
import struct
import base64

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

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

def parse_cif_file(file_path):
    """Parse a CIF file and return the image data"""
    
    with open(file_path, 'rb') as f:
        # Read header (16 bytes)
        header = f.read(16)
        version = struct.unpack('<I', header[0:4])[0]
        signature = header[4:8]
        width = struct.unpack('<I', header[8:12])[0]
        height = struct.unpack('<I', header[12:16])[0]
        
        print(f"Image format: CIF version {version}")
        print(f"Signature: {signature.hex()}")
        print(f"Dimensions: {width}x{height}")
        
        # Read color palette (768 bytes = 256 colors * 3 bytes)
        palette_data = f.read(0x300)
        
        # Process palette - right shift each byte by 2 as in the assembly code
        palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(256, 3)
        # palette = (palette >> 2).astype(np.uint8)  # Shift right by 2 bits

        if os.path.basename(file_path).startswith('F'):
            palette = adjust_palette(palette, 1.0, 1.0, 0)
        
        # Read the compressed image data
        compressed_data = f.read()
        
    # Decompress the LZW data
    image_data = decompress_lzw(compressed_data, width, height)
    
    if image_data is None or len(image_data) != width * height:
        print(f"Warning: Decompression may be incomplete. Expected {width*height} bytes, got {len(image_data) if image_data else 0}")
        # If decompression fails or is incomplete, use a placeholder
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
    
    return rgb_image

def display_cif(file_path):
    """Load and display a CIF image"""
    try:
        img_data = parse_cif_file(file_path)
        
        # Display using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(img_data)
        plt.title(f"CIF Image: {file_path}")
        plt.axis('off')
        plt.show()
        
        # You can also save as PNG if needed
        # pil_img = Image.fromarray(img_data)
        # output_path = file_path.rsplit('.', 1)[0] + '.png'
        # pil_img.save(output_path)
        # print(f"Image saved as {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def parse_cif_from_base64(base64_data):
    """Parse and display CIF from a base64 encoded string"""
    try:
        # Decode base64 data
        binary_data = base64.b64decode(base64_data)
        
        # Put the data in a BytesIO object to simulate a file
        file_like = io.BytesIO(binary_data)
        
        # Read header (16 bytes)
        header = file_like.read(16)
        version = struct.unpack('<I', header[0:4])[0]
        signature = header[4:8]
        width = struct.unpack('<I', header[8:12])[0]
        height = struct.unpack('<I', header[12:16])[0]
        
        print(f"Image format: CIF version {version}")
        print(f"Signature: {signature.hex()}")
        print(f"Dimensions: {width}x{height}")
        
        # Read color palette (768 bytes = 256 colors * 3 bytes)
        palette_data = file_like.read(0x300)
        
        # Process palette - right shift each byte by 2 as in the assembly code
        palette = np.frombuffer(palette_data, dtype=np.uint8).reshape(256, 3)
        palette = (palette >> 2).astype(np.uint8)  # Shift right by 2 bits
        
        # Read the compressed image data
        compressed_data = file_like.read()
        
        # Decompress the LZW data (implementation would depend on the specific LZW variant used)
        image_data = decompress_lzw(compressed_data, width, height)
        
        if image_data is None or len(image_data) != width * height:
            print(f"Warning: Decompression may be incomplete. Expected {width*height} bytes, got {len(image_data) if image_data else 0}")
            # If decompression fails or is incomplete, use a placeholder
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
        
        # Display using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title("CIF Image from Base64")
        plt.axis('off')
        plt.show()
        
        return rgb_image
        
    except Exception as e:
        print(f"Error processing base64 data: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        display_cif(sys.argv[1])
    else:
        print("Usage: python cif_parser.py <cif_file_path>")
        
        # Alternatively, you can use the base64 data directly:
        # base64_data = "AQAAAESpjTOAAgAA4AEAAAAAAAAACAAIAAAICAgAAAgACAgIAAgICBAAABAI..."
        # parse_cif_from_base64(base64_data)
