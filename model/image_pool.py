# copy of https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py
from torch import cat, unsqueeze
import random
class ImagePool():
    '''
    Image buffer to store previously generated images. We can update the discriminators with this
    buffer using a history of generated images rather than the ones produced by the latest generators.
    '''
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            # create an empty pool
            self.number_of_images = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        images_to_return = []

        for image in images:
            image = unsqueeze(image.data, dim=0)

            if self.number_of_images < self.pool_size:
                # if the buffer is not full, keep inserting current images into the buffer
                self.number_of_images += 1
                self.images.append(image)
                images_to_return.append(image)
            else:
                p = random.uniform(0, 1)
                # 50/50 chance the buffer will return a random image in the buffer and insert the current image, or
                # append the current image
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    temp = self.images[random_id].clone()
                    images_to_return.append(temp)
                else:
                    images_to_return.append(image)
        images_to_return = cat(images_to_return,0)
        return images_to_return