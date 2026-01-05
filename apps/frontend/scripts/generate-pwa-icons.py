#!/usr/bin/env python3
"""
Generate PWA icons for FloodSafe.
Creates PNG icons from the SVG favicon, or generates simple placeholder icons.
"""

import os
from pathlib import Path

def generate_icons_with_pillow():
    """Generate simple placeholder icons using Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed. Install with: pip install Pillow")
        return False

    public_dir = Path(__file__).parent.parent / "public"

    # Icon sizes to generate
    sizes = [
        ("pwa-192x192.png", 192),
        ("pwa-512x512.png", 512),
        ("apple-touch-icon.png", 180),
    ]

    # FloodSafe brand colors
    bg_color = (59, 130, 246)  # #3B82F6 - blue
    shield_color = (255, 255, 255)  # white
    wave_color = (96, 165, 250)  # #60A5FA - light blue
    drop_color = (30, 64, 175)  # #1E40AF - dark blue

    for filename, size in sizes:
        # Create image with blue background
        img = Image.new('RGB', (size, size), bg_color)
        draw = ImageDraw.Draw(img)

        # Calculate proportions
        center = size // 2
        shield_width = int(size * 0.6)
        shield_height = int(size * 0.7)

        # Draw shield (simplified as rounded rectangle)
        shield_left = center - shield_width // 2
        shield_top = int(size * 0.1)
        shield_right = center + shield_width // 2
        shield_bottom = shield_top + shield_height

        # Shield body (rectangle + triangle bottom)
        draw.rectangle(
            [shield_left, shield_top, shield_right, shield_bottom - shield_height // 4],
            fill=shield_color
        )
        # Shield point
        draw.polygon([
            (shield_left, shield_bottom - shield_height // 4),
            (shield_right, shield_bottom - shield_height // 4),
            (center, shield_bottom)
        ], fill=shield_color)

        # Draw wave pattern at bottom of shield
        wave_top = shield_bottom - shield_height // 3
        draw.rectangle(
            [shield_left + 10, wave_top, shield_right - 10, shield_bottom - 20],
            fill=wave_color
        )

        # Draw water drop in center
        drop_size = size // 8
        drop_top = shield_top + int(shield_height * 0.2)
        draw.ellipse(
            [center - drop_size, drop_top, center + drop_size, drop_top + drop_size * 2],
            fill=drop_color
        )

        # Save
        output_path = public_dir / filename
        img.save(output_path, "PNG")
        print(f"Created: {output_path}")

    return True


def generate_icons_with_cairosvg():
    """Generate icons from SVG using cairosvg."""
    try:
        import cairosvg
    except ImportError:
        print("cairosvg not installed. Falling back to Pillow...")
        return False

    public_dir = Path(__file__).parent.parent / "public"
    svg_path = public_dir / "favicon.svg"

    if not svg_path.exists():
        print(f"SVG not found: {svg_path}")
        return False

    sizes = [
        ("pwa-192x192.png", 192),
        ("pwa-512x512.png", 512),
        ("apple-touch-icon.png", 180),
    ]

    for filename, size in sizes:
        output_path = public_dir / filename
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            output_width=size,
            output_height=size
        )
        print(f"Created: {output_path}")

    return True


def main():
    print("Generating PWA icons for FloodSafe...")
    print("-" * 40)

    # Try cairosvg first (higher quality), fall back to Pillow
    if not generate_icons_with_cairosvg():
        if not generate_icons_with_pillow():
            print("\nFailed to generate icons. Please install either:")
            print("  pip install cairosvg")
            print("  pip install Pillow")
            return 1

    print("-" * 40)
    print("Done! Icons generated in apps/frontend/public/")
    return 0


if __name__ == "__main__":
    exit(main())
