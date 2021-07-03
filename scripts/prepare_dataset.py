import numpy as np

def get_mnist_batch_color(batch_raw, lena, change_colors=False):
    
    # Select random batch (WxHxC)
    batch_size = batch_raw.shape[0]
    img_size =  batch_raw.shape[2]
    
    # Resize
    batch_resized = batch_raw
    
    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=1)
    
    # Make binary
    batch_binary = (batch_rgb > -0.7)
    
    batch = np.zeros((batch_size, 3, img_size, img_size))
    
    for i in range(batch_size):
        # Take a random crop of the Lena image (background)

        x_c = np.random.randint(0, lena.size[0] - img_size)
        y_c = np.random.randint(0, lena.size[1] - img_size)
        image = lena.crop((x_c, y_c, x_c + img_size, y_c + img_size))
        image = np.asarray(image) / 255.0
        image = np.transpose(image, (2, 0, 1))

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number

        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        
        batch[i] = image
    
    # Map the whole batch to [-1, 1]
    #batch = batch / 0.5 - 1

    return batch