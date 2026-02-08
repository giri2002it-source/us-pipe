import os
from collections import Counter
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def pdf_to_images(pdf_path):
    poppler_path = r"C:\Users\Admin\PycharmProjects\PythonProject\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    return convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)


def extract_symbols_with_boxes(text, image):
    """Extract symbols AND get bounding boxes for visualization"""
    words = text.lower().split()
    common_words = {'the', 'and', 'for', 'are', 'you', 'with', 'this', 'from'}
    symbols = [w for w in words if len(w) > 2 and w not in common_words and w.isalnum()]

    # Get data with bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config='--psm 6')
    symbol_boxes = []

    for i in range(len(data['text'])):
        word = data['text'][i].lower().strip()
        if (len(word) > 2 and word not in common_words and word.isalnum() and
                int(data['conf'][i]) > 30):  # Confidence > 30%
            symbol_boxes.append({
                'text': word,
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'conf': data['conf'][i]
            })
    return Counter([box['text'] for box in symbol_boxes]), symbol_boxes


def show_image_with_symbols(image, symbol_boxes, title, save_path=None):
    """Display image with bounding boxes around detected symbols"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')

    for box in symbol_boxes:
        rect = patches.Rectangle((box['x'], box['y']), box['w'], box['h'],
                                 linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(box['x'], box['y'] - 5, box['text'], color='red', fontsize=8, fontweight='bold')

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    STATIC_PDF = "Revised_Print.pdf"

    if not os.path.exists(STATIC_PDF):
        print(f"❌ Put '{STATIC_PDF}' in project folder!")
        return

    print("📄 Processing ALL pages for symbols...")
    pages = pdf_to_images(STATIC_PDF)
    print(f"✅ Total {len(pages)} pages loaded!")

    # 1) EXTRACT ALL SYMBOLS FROM ALL PAGES (REFERENCE)
    all_reference_symbols = Counter()
    all_symbol_boxes = {}  # Store boxes per page

    print("\n🔍 Extracting ALL symbols from ALL pages...")
    for i, page in enumerate(pages):
        print(f"  Processing page {i + 1}...")
        gray_page = page.convert('L')
        text = pytesseract.image_to_string(gray_page, config='--psm 6').lower()
        symbols, boxes = extract_symbols_with_boxes(text, gray_page)

        all_reference_symbols.update(symbols)
        all_symbol_boxes[i + 1] = boxes

    print("\n📋 ALL REFERENCE SYMBOLS (from ALL pages):")
    for symbol, count in all_reference_symbols.most_common(20):
        print(f"  {symbol:<12} | {count:3d}")

    # 2) PAGE 2 BLUEPRINT MATCHING
    if len(pages) < 2:
        print("❌ Need at least 2 pages!")
        return

    print("\n🔍 Page 2 - Blueprint matching against ALL references...")
    gray_page2 = pages[1].convert('L')
    text_page2 = pytesseract.image_to_string(gray_page2, config='--psm 6').lower()
    blueprint_symbols, blueprint_boxes = extract_symbols_with_boxes(text_page2, gray_page2)

    # MATCH COUNTS
    print("\n🎯 MATCHED SYMBOLS IN BLUEPRINT (Page 2):")
    matches = Counter()
    total_matches = 0

    for symbol, ref_count in all_reference_symbols.most_common():
        if symbol in blueprint_symbols:
            match_count = min(ref_count, blueprint_symbols[symbol])
            matches[symbol] = match_count
            total_matches += match_count
            print(f"✅ {symbol:<12} | Ref:{ref_count:3d} | Found:{match_count:2d} | Confidence: OK")

    # SUMMARY
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Total unique reference symbols:  {len(all_reference_symbols)}")
    print(f"   Blueprint symbols (Page 2):      {len(blueprint_symbols)}")
    print(f"   ✅ Total matches:                {total_matches}")
    print(f"   ✅ Match rate:                  {total_matches / len(all_reference_symbols) * 100:.1f}%")

    # 3) VISUALIZE with images
    print("\n🖼️  Creating visualizations...")

    # Show Page 1 with symbols
    show_image_with_symbols(pages[0].convert('L'), all_symbol_boxes[1],
                            f'Page 1 - ALL Reference Symbols Detected', 'page1_symbols.png')

    # Show Page 2 blueprint with matches
    show_image_with_symbols(gray_page2, blueprint_boxes,
                            f'Page 2 - Blueprint (Matched Symbols)', 'page2_blueprint.png')


if __name__ == "__main__":
    main()
