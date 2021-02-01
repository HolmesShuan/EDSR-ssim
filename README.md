# EDSR-ssim
Different SSIM metrics in CNN-based super resolution algorithms, such as [EDSR](https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba), [RDN](https://github.com/yulunzhang/RDN) and [MSRN](https://github.com/MIVRC/MSRN-PyTorch).
We transform those MATLAB codes to Python. 

### How to use ?
1. Please refer to the [PSNR setting](https://github.com/thstkdgus35/EDSR-PyTorch/blob/a90b54d1b7f8ee780920e2ccd30692ab8e783f46/src/utility.py#L165) in EDSR, edit `utility.py` as follow:
```Python
'''
def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
'''
import numpy as np
from scipy import signal
from skimage.measure import compare_ssim

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def calc_ssim(X, Y, scale, rgb_range, dataset=None, sigma=1.5, K1=0.01, K2=0.03, R=255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
  Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
  The authors of EDSR use MATLAB's ssim as the evaluation tool, 
  thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  if dataset and dataset.dataset.benchmark:
    shave = scale
    if X.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        X = X.mul(convert).sum(dim=1)
        Y = Y.mul(convert).sum(dim=1)
  else:
    shave = scale + 6

  X = X[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64) 
  Y = Y[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)

  window = gaussian_filter

  ux = signal.convolve2d(X, window, mode='same', boundary='symm')
  uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

  uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
  uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
  uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  return mssim
```
2. Edit [Line 93](https://github.com/thstkdgus35/EDSR-PyTorch/blob/a90b54d1b7f8ee780920e2ccd30692ab8e783f46/src/trainer.py#L93) in `trainer.py` as follow:

```Python
'''
self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
    sr, hr, scale, self.args.rgb_range, dataset=d
)
'''
self.ckp.log[-1, idx_data, idx_scale] += utility.calc_ssim(
    sr, hr, scale, self.args.rgb_range, dataset=d
)
```

### Other SSIMs:

Different SSIM metrics used in Super-Resolution papers:

```python
import numpy as np
from scipy import signal
from skimage.measure import compare_ssim

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def calc_ssim(X, Y, sigma=1.5, K1=0.01, K2=0.03, R = 255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of Evaluate_PSNR_SSIM.m in RDN (Residual Dense Network for Image Super-Resolution CVPR2018).
  Official Link : https://github.com/yulunzhang/RDN
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  X = X.astype(np.float64)
  Y = Y.astype(np.float64)

  # Since matlab_style_gauss2D() yields normalized filter, this operation can be deprecated.
  window = gaussian_filter / np.sum(np.sum(gaussian_filter))
  
  window = np.fliplr(window)
  window = np.flipud(window)

  ux = signal.convolve2d(X, window, mode='valid', boundary='fill', fillvalue=0)
  uy = signal.convolve2d(Y, window, mode='valid', boundary='fill', fillvalue=0)

  uxx = signal.convolve2d(X*X, window, mode='valid', boundary='fill', fillvalue=0)
  uyy = signal.convolve2d(Y*Y, window, mode='valid', boundary='fill', fillvalue=0)
  uxy = signal.convolve2d(X*Y, window, mode='valid', boundary='fill', fillvalue=0)

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  return mssim

def calc_ssim(X, Y, sigma=1.5, K1=0.01, K2=0.03, R=255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
  Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
  The authors of EDSR use MATLAB's ssim as the evaluation tool, 
  thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  X = X.astype(np.float64)
  Y = Y.astype(np.float64)

  window = gaussian_filter

  ux = signal.convolve2d(X, window, mode='same', boundary='symm')
  uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

  uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
  uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
  uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  return mssim

def calc_ssim(X, Y):
  '''
  X (groundtruth): y channel (i.e., luminance) of transformed YCbCr space of X
  Y (prediction): y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of test.py in MSRN (Multi-scale Residual Network for Image Super-Resolution ECCV2018).
  Official Link : https://github.com/MIVRC/MSRN-PyTorch
  The authors of MSRN use scikit-image's compare_ssim as the evaluation tool, 
  note that this function is quite sensitive to the argument "data_range", emprically, the larger the higher output.
  '''
  ssim = compare_ssim(X, Y, data_range=max(Y.max(),X.max()) - min(X.min(),Y.min()) # one may obtain a slightly higher output than original setting
  # ssim = compare_ssim(X, Y, data_range=Y.max() - X.min())
  return ssim
```
**Note that** we omit the crop preprocess, which can be a game changer for super resolution problems. 
Please follow the exact crop setting in those papers before calling SSIM functions.

### Disclaimer:
For better or worse performances than reported results in those papers, we are not responsible. Hopefully, you may find these codes helpful in your research or work.
