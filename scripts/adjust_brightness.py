import cv2
import numpy as np
import argparse

def adjust_brightness(image, brightness):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + float(brightness), 0, 255)
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

def main(input_image_path, output_image_path, brightness):

    image = cv2.imread(input_image_path)
    for i in range (-brightness, brightness, 10):
        adjusted_image = adjust_brightness(image, float(i))
        cv2.imwrite(output_image_path, adjusted_image)
        cv2.imshow("image", adjusted_image)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust image brightness.')
    parser.add_argument('--input', type=str, required=True, help='')
    parser.add_argument('--output', type=str, required=True, help='')
    parser.add_argument('--brightness', type=int, required=True, help='positive or negative')
    args = parser.parse_args()

    main(args.input, args.output, args.brightness)
