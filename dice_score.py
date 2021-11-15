from PIL import Image
import numpy as np

def compute_dice(label_img, pred_img, p_threshold=0.5):
    p = pred_img.astype(np.float32)
    l = label_img.astype(np.float32)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > p_threshold] = 1.0
    p[p < p_threshold] = 0.0
    l[l > p_threshold] = 1.0
    l[l < p_threshold] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val

#input_path = '/home/jsy/font/new_font_120_500_1000_SR/fake_samples-154-450_train.png'
input_path = '/home/jsy/font/son_img/fake_samples-74-1050_train.png'
### original base-GAN
#input_path = '/home/jsy/font/son_fontGAN/fake_samples-261-100_train.png'
#input_path = '/home/jsy/font/son_img/fake_samples-74-1050_train.png'
### fine-tuning base-GAN

#target_path = '/home/jsy/font/fixed_set/t_fixed_target2.png'
target_path = '/home/jsy/font/fixed_set/t_fixed_target.png'
a = Image.open(input_path)
#a = (np.ones((262, 1042, 3)) * 255).astype(np.uint8)

# rand = np.random.rand(262, 1042, 3)
# a = (rand * 255).astype(np.uint8)
# a = (np.zeros((262, 1042, 3))).astype(np.uint8)
a = np.asarray(a)

b = Image.open(target_path)
b = np.asarray(b)

score = compute_dice(a, b)
print(score)