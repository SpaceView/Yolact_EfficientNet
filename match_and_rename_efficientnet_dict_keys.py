import torch
from collections import OrderedDict

# you can define a complicated function to do some thing as below
def key_transformation(key):
    print(key)
    return key

# efficient NET
#src_dir = './weights/efficientnet-b0-355c32eb.pth'
#dst_dir = './weights/efficientnet-b0-sav.pth'

#src_dir = './weights/efficientdet-d0.pth'
#dst_dir = './weights/efficientdet-d0-sav.pth'
#f = open('./weights/state_dict_name.txt', 'w')

yolact_dir = './weightsav/yolact_EfficientNet_0_100.pth'
effnet_dir = './weights/efficientnet-b0-355c32eb.pth'
save_dir = './weightsav/efficientnet-b0-yolact.pth'
f_eff = open('./weightsav/efficientnet-b0_keys.txt', 'w')
f_yol = open('./weightsav/yolact_EfficientNet_bacbone_keys.txt', 'w')

yolact_backbone_keys = []
effnet_keys = []
new_state_dict = OrderedDict()

if  __name__ == '__main__':
    yolact_state_dict = torch.load(yolact_dir)

    for yol_key, yol_value in yolact_state_dict.items():
        if yol_key.startswith('backbone.model.'):
            new_key = yol_key.replace('backbone.model.', '')
            yolact_backbone_keys.append(new_key)
            f_yol.write(new_key)
            f_yol.write('\n')
    f_yol.close()

    effnet_state_dict = torch.load(effnet_dir)
    index = 0
    YOL_NUM = len(yolact_backbone_keys)
    for eff_key, eff_value in effnet_state_dict.items():
        effnet_keys.append(eff_key)
        f_eff.write(eff_key)
        f_eff.write('\n')
        if(index < YOL_NUM):
            yol_key = yolact_backbone_keys[index]
            if(eff_key != yol_key):
                tkey = yol_key.replace('.conv.', '.')
                assert(tkey == eff_key)
            new_state_dict[yol_key] = eff_value
            index = index + 1
        else:
            new_state_dict[eff_key] = eff_value
    f_eff.close()

    torch.save(new_state_dict, save_dir)

    print('done')