import argparse
import operator
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='ML HW0')
    parser.add_argument('f1', type=str, help='file1')
    parser.add_argument('f2', type=str, help='file2')
    parser.add_argument('out', type=str, help='out')
    args = parser.parse_args()

    im1 = Image.open(args.f1)
    pixels1 = im1.load()
    im2 = Image.open(args.f2)
    pixels2 = im2.load()

    if im1.size != im2.size:
        print('warn: images with different size')

    out = Image.new('RGBA', im1.size)
    pixels_out = out.load()
    
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):            
            if pixels1[i, j] == pixels2[i, j]:
                pixels_out[i, j] = (0, 0, 0, 0)
            else:
                pixels_out[i, j] = pixels2[i, j]
    
    out.save(args.out)
    
if __name__ == '__main__':
    main()
