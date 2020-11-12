import unet
import segnet
model_from_name = {}



model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
model_from_name["resnet50_unet"] = unet.resnet50_unet



model_from_name["segnet"] = segnet.segnet


