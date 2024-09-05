import numpy as np
import torch
import cv2

def calculate_ssim(img1, img2, scale=2):
    img1 = img1.cpu().detach().data.numpy()
    img2 = img2.cpu().detach().data.numpy()

    img1 = np.transpose(img1, (2,3,1,0)) * 255.
    img2 = np.transpose(img2, (2,3,1,0)) * 255.

    if not img1.shape == img2.shape:
        print('[ERRO] images must have the same dimenstions')
        assert(0)

    ssims = []

    for b in range(img1.shape[3]):
        img1_y = img1[scale:-scale, scale:-scale, :, b]
        img2_y = img2[scale:-scale, scale:-scale, :, b]

        if   img1_y.shape[2] == 1:
            img1_y = img1_y.squeeze()
            img2_y = img2_y.squeeze()

        elif img1_y.shape[2] == 3:
            img1_y = np.dot(img1_y, [65.738, 129.057, 25.064]) / 255. + 16.0
            img2_y = np.dot(img2_y, [65.738, 129.057, 25.064]) / 255. + 16.0

        ssims.append(torch.as_tensor(ssim(img1_y, img2_y)))

    return torch.as_tensor(ssims)

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()