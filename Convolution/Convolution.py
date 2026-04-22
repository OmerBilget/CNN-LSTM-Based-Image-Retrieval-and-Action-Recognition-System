import numpy as np
import matplotlib.pyplot as plt
import cv2


#Load Image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found")

image = image.astype(np.float32) / 255.0  # normalize 0-1 range

# Create Filters
def gaussian_kernel(size=7, sigma=1.5):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

edge = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

blur = gaussian_kernel(9, 2.0).astype(np.float32)

filters = [edge, blur, sharpen]
filter_names = ["Edge (3x3)", "Gaussian Blur (9x9)", "Sharpen (3x3)"]


# Padding Function
def apply_padding(image, padding):
    if padding == 0:
        return image
    h, w = image.shape
    padded = np.zeros((h + 2*padding, w + 2*padding), dtype=image.dtype)
    padded[padding:padding+h, padding:padding+w] = image
    return padded


# Single Convolution
def convolve_single(image, kernel, stride, padding):
    kernel_h, kernel_w = kernel.shape

    padded = apply_padding(image, padding)
    padded_h, padded_w = padded.shape

    out_h = (padded_h - kernel_h) // stride + 1
    out_w = (padded_w - kernel_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = padded[
                i*stride : i*stride + kernel_h,
                j*stride : j*stride + kernel_w
            ]
            output[i, j] = np.sum(region * kernel)

    return output

# Multi-filter Convolution
def convolve(image, filters, stride=1, padding=0):
    outputs = []
    for kernel in filters:
        fm = convolve_single(image, kernel, stride, padding)
        outputs.append(fm)
    return outputs


# 6. Run Different padding and stride Settings
configs = [
    (0, 1),
    (1, 1),
    (2, 1),
    (1, 2)
]

results = []

for padding, stride in configs:
    fmaps = convolve(image, filters, stride=stride, padding=padding)
    results.append((padding, stride, fmaps))


# 7. Normalize for display
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# 8. Visualization
rows = len(results)
cols = len(filters) + 1

plt.figure(figsize=(4 * cols, 3 * rows))

for r, (p, s, fmaps) in enumerate(results):

    # Original
    plt.subplot(rows, cols, r*cols + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original\nP={p}, S={s}")
    plt.axis('off')

    # Filters output
    for c, fm in enumerate(fmaps):
        plt.subplot(rows, cols, r*cols + c + 2)
        plt.imshow(normalize(fm), cmap='gray')
        plt.title(filter_names[c])
        plt.axis('off')

plt.tight_layout()
plt.show()