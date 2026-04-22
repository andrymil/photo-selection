import cv2
import os
import shutil
import glob

SOURCE_DIR = "datasets/CUHK/image"
SHARP_DIR = "datasets/CUHK/sharp"
BLUR_DIR = "datasets/CUHK/blurred"

os.makedirs(SHARP_DIR, exist_ok=True)
os.makedirs(BLUR_DIR, exist_ok=True)


def main():
    image_paths = glob.glob(f"{SOURCE_DIR}/motion*.jpg")
    print(f"Found {len(image_paths)} motion images to label.")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        height, width = img.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        cv2.imshow(
            "Tinder for Dataset - 'a': Sharp | 'd': Blur | 'q': Quit",
            img,
        )

        filename = os.path.basename(path)

        valid_key_pressed = False
        while not valid_key_pressed:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("a"):
                shutil.move(path, os.path.join(SHARP_DIR, filename))
                print(f"[{filename}] -> SHARP")
                valid_key_pressed = True
            elif key == ord("d"):
                shutil.move(path, os.path.join(BLUR_DIR, filename))
                print(f"[{filename}] -> BLUR")
                valid_key_pressed = True
            elif key == ord("q"):
                print("Labeling aborted by user.")
                cv2.destroyAllWindows()
                return
            else:
                print("Wrong key! Use 'a', 'd' or 'q'. Try again.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
