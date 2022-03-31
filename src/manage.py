import click
from cv2 import imwrite, imread, IMREAD_COLOR, IMREAD_GRAYSCALE
from gradient_finder.gradient import gradient_positions, draw_rectangle
from preprocessing.sobel import greyscale_sobel
import os


@click.command()
@click.option('--strict', is_flag=True)
@click.argument('out')
@click.argument('img')
def main(strict, out, img):
    if not os.path.exists(img):
        print(f'{img} not found')
        return

    img_preprocessed = greyscale_sobel(img)
    grayscale_img = imread(img, IMREAD_GRAYSCALE)
    color_img = imread(img, IMREAD_COLOR)

    frame = gradient_positions(img_preprocessed, grayscale_img, strict)
    color_img = draw_rectangle(
        color_img,
        frame,
        (255, 0, 0),
        3
    )
    imwrite(out, color_img)


if __name__ == '__main__':
    main()
