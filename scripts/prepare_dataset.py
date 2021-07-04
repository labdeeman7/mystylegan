import numpy as np

class get_color_mnist(object):
  def __init__(self, template_img):
        self.template_img = template_img

  def __call__(self, sample):
    # Select random batch (WxHxC)
    img_size =  sample.shape[1]
    template_img = self.template_img
    
    # Extend to RGB
    img_rgb = np.concatenate([sample, sample, sample], axis=0)
    
    # Make binary
    img_binary = (img_rgb > -0.7)
    
    x_c = np.random.randint(0, template_img.size[0] - img_size)
    y_c = np.random.randint(0, template_img.size[1] - img_size)
    image = template_img.crop((x_c, y_c, x_c + img_size, y_c + img_size))
    image = np.asarray(image) / 255.0
    image = np.transpose(image, (2, 0, 1))

    # Invert the colors at the location of the number
    image[img_binary] = 1 - image[img_binary]

    return image