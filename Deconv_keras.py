from DeconvLibrary import *


parser = argparser()
args = parser.parse_args()
image_path = args.image
print("hahaha")
layer_name = args.layer_name
feature_to_visualize = args.feature
visualize_mode = args.mode

model = vgg16.VGG16(weights='imagenet', include_top=True)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
if not layer_dict.has_key(layer_name):
    print('Wrong layer name')
    sys.exit()

# Load data and preprocess
img = Image.open(image_path)
img = img.resize(224, 224)
img_array = np.array(img)
img_array = np.transpose(img_array, (2, 0, 1))
img_array = img_array[np.newaxis, :]
img_array = img_array.astype(np.float)
img_array = imagenet_utils.preprocess_input(img_array)

deconv = visualize(model, img_array,
                   layer_name, feature_to_visualize, visualize_mode)

# postprocess and save image
deconv = np.transpose(deconv, (1, 2, 0))
deconv = deconv - deconv.min()
deconv *= 1.0 / (deconv.max() + 1e-8)
deconv = deconv[:, :, ::-1]
uint8_deconv = (deconv * 255).astype(np.uint8)
img = Image.fromarray(uint8_deconv, 'RGB')
img.save('results/{}_{}_{}.png'.format(layer_name, feature_to_visualize, visualize_mode))