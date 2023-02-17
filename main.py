import cv2


def import_image_resize(file_name:str, width:int, heigth:int):
    return cv2.resize(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE), (width, heigth))

if __name__ == '__main__':
    img_depth = import_image_resize("input.jpg", 256, 256)
    cv2.imshow("Depth Image from Kinect", img_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("hi")