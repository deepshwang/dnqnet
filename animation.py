import numpy as np
from e2cnn.nn import *
from e2cnn.group import *
from e2cnn.gspaces import *

from models.models import *
from models.cnn import CNN
from params import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

import matplotlib.animation as manimation

from skimage.transform import resize
import scipy.ndimage

import torch
from typing import Union
import pdb
import imageio
import cv2

plt.rcParams['image.cmap'] = 'hot'
plt.rcParams['axes.titlepad'] = 30

# the irrep of frequency 1 of SO(2) produces the usual 2x2 rotation matrices
rot_matrix = SO2(1).irrep(1)




def build_mask(s: int, margin: float = 2., dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s - 1) / 2
    t = (c - margin / 100. * c) ** 2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = np.exp((t - r) / sig ** 2)
            else:
                mask[..., x, y] = 1.
    return mask


def domask(x: Union[np.ndarray, torch.Tensor], margin=2, fmt="torch"):
    if fmt == "image":
        s = x.shape[0]
        mask = build_mask(s, margin)
        mask = mask.permute(0, 2, 3, 1).squeeze()
    else:
        s = x.shape[2]
        mask = build_mask(s, margin)

    if isinstance(x, np.ndarray):
        mask = mask.numpy()

    # use an inverse mask to create a white background (value = 1) instead of a black background (value = 0)
    return mask * x + 1. - mask


def animate(model: EquivariantModule,
            cnnmodel,
            image: Union[str, np.ndarray],
            image2: Union[str, np.ndarray],
            outfile: str,
            drawer: callable,
            R: int = 72,
            S: int = 71,
            duration: float = 10.,
            figsize=(21, 10),
            args=None
            ):
    r'''

    Build a video animation

    Args:
        model: the equivariant model
        image: the input image
        outfile: name of the output file
        drawer: method which plots the output field. use one of the methods ``draw_scalar_field``, ``draw_vector_field`` or ``draw_mixed_field``
        R: number of rotations of the input to render, i.e. number of frames in the video
        S: size the input image is downsampled to before being fed in the model
        duration: duration (in seconds) of the video
        figsize: shape of the video (see matplotlib.pyplot.figure())


    '''

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=R / duration, metadata=metadata)
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    fig.set_tight_layout(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #CHANGEHERE
    # model = cnnmodel
    model.to(device)
    model.eval()

    if isinstance(image, str):
        image = mpimg.imread(image).transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    _, C, w, h = image.shape

    if isinstance(image2, str):
        image2 = mpimg.imread(image2).transpose((2, 0, 1))
    image2 = image2[np.newaxis, :, :, :]

    # resize the image to have a squared shape
    # the image is initially upsampled to (up to) 4 times the specified size S
    # rotations are performed at this higher resolution and are later downsampled to size S
    # this helps reducing the interpolation artifacts for rotations which are not multiple of pi/2
    T = max(4 * S + 1, 513)
    image = resize(image, (1, C, T, T), anti_aliasing=True)
    image2 = resize(image2, (1, C, T, T), anti_aliasing=True)


    print('Image Loaded')

    original_inputs = []
    original_inputs2 = []


    for r in range(R):
        print(f"{r}/{R}")

        # Rotate the image
        # N.B.: this only works for trivial (i.e. scalar) input fields like RGB images.
        # In case vector fields are used in input, one should also rotate the channels using the group representation
        # of the corresponding FieldType
        rot_input = scipy.ndimage.rotate(image, r * 360.0 / R, (-2, -1), reshape=False, order=2)
        rot_input2 = scipy.ndimage.rotate(image2, r * 360.0 / R, (-2, -1), reshape=False, order=2)

        # discard non-RGB channels
        rot_input = rot_input[:, :3, ...]
        rot_input2 = rot_input2[:, :3, ...]

        original_inputs.append(rot_input)
        original_inputs2.append(rot_input2)

    original_inputs = np.concatenate(original_inputs, axis=0)
    original_inputs2 = np.concatenate(original_inputs2, axis=0)



    # mask the input images to remove the pixels which would be moved outside the grid by a rotation
    original_inputs *= build_mask(T, margin=5).numpy()
    #
    # for i, layer in enumerate(original_inputs):
    #     viz = layer.transpose(1, 2, 0)
    #     imageio.imwrite("./test/" + str(i)+"_th_inputmask.png", viz)

    # downsample the images
    inputs = resize(original_inputs, (original_inputs.shape[0], C, S, S), anti_aliasing=True)
    inputs2 = resize(original_inputs2, (original_inputs2.shape[0], C, S, S), anti_aliasing=True)


    rotated_input = torch.tensor(inputs, dtype=torch.float32)
    rotated_input2 = torch.tensor(inputs2, dtype=torch.float32)
    rotated_input *= build_mask(S, margin=5.2)

    # normalize the colors of the images before feeding them into the model
    rotated_input -= rotated_input[0, ...].view(3, -1).mean(dim=1).view(1, 3, 1, 1)
    rotated_input /= rotated_input[0, ...].view(3, -1).std(dim=1).view(1, 3, 1, 1)

    rotated_input2 -= rotated_input2[0, ...].view(3, -1).mean(dim=1).view(1, 3, 1, 1)
    rotated_input2 /= rotated_input2[0, ...].view(3, -1).std(dim=1).view(1, 3, 1, 1)

    del inputs

    rotated_input = rotated_input.to(device)
    rotated_input2 = rotated_input2.to(device)
    # wrap the tensor in a GeometricTensor

    if args.viz_network == 'E2CNN':
        rotated_input = GeometricTensor(rotated_input, model.in_type)
        rotated_input2 = GeometricTensor(rotated_input2, model.in_type)


    # pass the images through the model to compute the output field
    with torch.no_grad():

        # In training mode, the batch normalization layers normalize the features with the batch statistics
        # This sometimes produces nicer output fields
        # model.train()

        output = model(rotated_input)
        output2 = model(rotated_input2)

    ### Invariant Space Visualization

    # a = output[2]
    # b = output2[2]
    #
    # a_fixed = output[2][0]
    #
    # a_inter_l2=[]
    # a_exter_l2=[]
    # a_l2=[]
    # for a_rot in a:
    #     a_inter_l2.append(torch.norm(a_fixed-a_rot).item())
    #     a_l2.append(torch.norm(a_rot).item())
    # for b_rot in b:
    #     a_exter_l2.append(torch.norm(a_fixed-b_rot).item())
    #
    # plt2.title("")
    # plt2.plot(list(range(0, 360, 5)), a_inter_l2, 'r-s')
    # plt2.plot(list(range(0, 360, 5)), a_exter_l2, 'b-o')
    # plt2.plot(list(range(0, 360, 5)), a_l2, 'g-o')
    # plt2.ylim(0, 8.7)
    # plt2.legend(["Rotations of the same image", "Rotations of a different image", "L2 size of rotated image"])
    # plt2.xlabel("Rotated Degrees")
    # plt2.ylabel("L2 distance of invariant feature")
    # plt2.savefig("./test.png")
    # pdb.set_trace()





    # extract the underlying torch.Tensor
    # if args.viz_network == 'E2CNN':
    #     output = output.tensor
    #     output2 = output2.tensor


    # output = output[1][:,:3,:,:]
    # output2 = output2[1][:,:3,:,:]
    output = output[:,:3,:,:]
    output2 = output2[:,:3,:,:]

    output = output.cpu()
    output = output.detach()
    output = output.numpy().transpose(0, 2, 3, 1)
    # for i, layer in enumerate(output):
    #     viz = layer[:, :, :3]
    #     imageio.imwrite(str(i)+"_th_output_withnoinputmask.png", viz)

    output = output[:,: ,:, 0]




    # mask the inputs with a white background for visualization purpose
    original_inputs = domask(original_inputs, margin=5)

    # visualize each rotated image and its corresponding output in a different frame of the video
    with writer.saving(fig, outfile, 100):
        for i, r in enumerate(range(R)):
            print(f"{r}/{R}")

            # render the input image
            # axs[0].clear()
            # axs[0].imshow(original_inputs[r, ...].transpose(1, 2, 0))
            # axs[0].set_title("input", fontdict={'fontsize': 30})

            # render the output and the stabilized output
            # if r > 0 and r % 2 == 0:
            drawer(axs[1:], output, r, i)

            # for ax in axs:
            #     ax.axis('off')
            #
            # fig.set_tight_layout(True)
            # plt.draw()
            #
            # writer.grab_frame()


def draw_scalar_field(axs, scalarfields, r: int, i):
    '''
    Draw a scalar field
    '''

    D = 3


    m, M = scalarfields.min(), scalarfields.max()

    R = scalarfields.shape[0]
    angle = r * 2 * np.pi / R
    angle_prev = (r-2) * 2 * np.pi / R
    scalarfield = scalarfields[r, ...].squeeze()
    scalarfield_prev = scalarfields[r-2, ...].squeeze()

    axs[0].clear()
    # sf = axs[0].imshow(scalarfield.repeat(D, axis=0).repeat(D, axis=1))
    sf = axs[0].imshow(domask(scalarfield.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"))
    axs[0].set_title("feature map", fontdict={'fontsize': 30})
    # sf.set_clim(m, M)

    stable_view = scipy.ndimage.rotate(scalarfield, -angle * 180.0 / np.pi, (-2, -1), reshape=False, order=2)
    # stable_view_prev = scipy.ndimage.rotate(scalarfield_prev, -angle_prev * 180.0 / np.pi, (-2, -1), reshape=False, order=2)
    # stable_view_diff = np.abs(stable_view-stable_view_prev)

    ### Save scalar field and stabilized view
    scalarfield_save = cv2.applyColorMap((domask(scalarfield/M*255, margin=8, fmt="image")).astype(np.uint8), cv2.COLORMAP_HSV)
    stable_view_save = cv2.applyColorMap((domask(stable_view/M*255, margin=8, fmt="image")).astype(np.uint8), cv2.COLORMAP_HSV)
    # stable_view_diff_save = cv2.applyColorMap((domask(stable_view_diff, margin=8, fmt="image")*255).astype(np.uint8), cv2.COLORMAP_HSV)
    imageio.imwrite('./test/featspace_'+str(i)+".jpg", scalarfield_save)
    imageio.imwrite('./test/featspace_stabilized_'+str(i)+".jpg", stable_view_save)


    axs[1].clear()
    # sf = axs[1].imshow(stable_view.repeat(D, axis=0).repeat(D, axis=1))
    sf = axs[1].imshow(domask(stable_view.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"))
    axs[1].set_title("stabilized view", fontdict={'fontsize': 30})
    sf.set_clim(m, M)


def draw_vector_field(axs, vectorfield, r: int):
    r'''
    Draw a vector field
    '''

    D = 21
    extent = 0, vectorfield.shape[1], 0, vectorfield.shape[2]

    mask = build_mask(D * vectorfield.shape[1], margin=2).numpy().transpose(0, 2, 3, 1).squeeze()

    R = vectorfield.shape[0]
    angle = r * 2 * np.pi / R

    norms = np.sqrt((vectorfield ** 2).sum(axis=3))
    m, M = norms.min(), norms.max()

    vectorfield = vectorfield[r, ...]
    norms = norms[r, ...]

    X = range(D // 2, D * extent[1], D)
    Y = range(D // 2, D * extent[3], D)

    submask = mask[D // 2:D * extent[1]:D, D // 2:D * extent[3]:D]

    axs[0].clear()
    sf = axs[0].imshow(domask(norms.repeat(D, axis=0).repeat(D, axis=1), fmt='image'))
    sf.set_clim(m, M)
    vf = axs[0].quiver(X, Y, vectorfield[:, :, 0] * submask, vectorfield[:, :, 1] * submask, color="green", units="xy",
                       width=1)
    axs[0].set_title("feature field", fontdict={'fontsize': 30})

    stable_view = scipy.ndimage.rotate(vectorfield, -angle * 180.0 / np.pi, (-3, -2), reshape=False, order=2)

    rm = rot_matrix(-angle)
    stable_view = np.einsum("oc,xyc->xyo", rm, stable_view)
    stable_norms = np.sqrt((stable_view ** 2).sum(axis=2))

    axs[1].clear()
    sf = axs[1].imshow(domask(stable_norms.repeat(D, axis=0).repeat(D, axis=1), fmt='image'))
    sf.set_clim(m, M)
    vf = axs[1].quiver(Y, X, stable_view[:, :, 0] * submask, stable_view[:, :, 1] * submask, color="green", units='xy',
                       width=1)
    axs[1].set_title("stabilized view", fontdict={'fontsize': 30})


def quiver(ax, X, Y, U, V):
    scale = 1. / 20.
    X, Y = np.meshgrid(X, Y)
    mask = V ** 2 + U ** 2 > 1e-3
    ax.quiver(X[mask], Y[mask], U[mask], V[mask], color="forestgreen", angles='xy', units="xy", scale=scale, width=1.3)


def draw_mixed_field(axs, featurefield, r):
    r'''
    Draw a field containing a scalar field and a vector field
    '''

    D = 3
    V = 3

    extent = 0, D * featurefield.shape[1], 0, D * featurefield.shape[2]

    mask = build_mask(featurefield.shape[1], margin=8).numpy().transpose(0, 2, 3, 1).squeeze()

    R = featurefield.shape[0]
    angle = r * 2 * np.pi / R

    scalarfield = featurefield[:, ..., 0]
    m, M = scalarfield.min(), scalarfield.max()

    vectorfield = featurefield[r, ..., 1:]
    scalarfield = featurefield[r, ..., 0]
    featurefield = featurefield[r, ...]

    X = range(V * D // 2, extent[1], V * D)
    Y = range(V * D // 2, extent[3], V * D)

    submask = mask[V // 2:extent[1]:V, V // 2:extent[3]:V]

    axs[0].clear()
    sf = axs[0].imshow(domask(scalarfield.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"))
    sf.set_clim(m, M)
    quiver(axs[0],
           X, Y,
           vectorfield[V // 2:extent[1]:V, V // 2:extent[3]:V, 0] * submask,
           vectorfield[V // 2:extent[1]:V, V // 2:extent[3]:V, 1] * submask,
           )
    axs[0].set_title("feature fields", fontdict={'fontsize': 30})

    stable_view = scipy.ndimage.rotate(featurefield, -angle * 180.0 / np.pi, (-3, -2), reshape=False, order=2)

    stable_vectorfield = stable_view[..., 1:]
    stable_scalarfield = stable_view[..., 0]

    rm = rot_matrix(-angle)
    stable_vectorfield = np.einsum("oc,xyc->xyo", rm, stable_vectorfield)

    axs[1].clear()
    sf = axs[1].imshow(domask(stable_scalarfield.repeat(D, axis=0).repeat(D, axis=1), margin=8, fmt="image"))
    sf.set_clim(m, M)
    quiver(axs[1],
           X, Y,
           stable_vectorfield[V // 2:extent[1]:V, V // 2:extent[3]:V, 0] * submask,
           stable_vectorfield[V // 2:extent[1]:V, V // 2:extent[3]:V, 1] * submask,
           )
    axs[1].set_title("stabilized view", fontdict={'fontsize': 30})

def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.8)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def build_gcnn(N: int, output: str):
    r'''
    Build an encoder-decoder model equivariant to N rotations.
    ``output`` speicifies the type of output field of the model, which will then be used for the animation.
    '''

    # build the g-space for N rotations
    if N == 1:
        gc = TrivialOnR2()
    else:
        gc = Rot2dOnR2(N)

    # the input contains 3 scalar channels (RGB colors)
    r1 = FieldType(gc, [gc.trivial_repr] * 3)

    # let's build a few inner layers
    # we will build a small encoder-decoder convolutional architecture
    layers = []

    r2 = FieldType(gc, [gc.regular_repr] * 8)
    cl1 = R2Conv(r1, r2, 5, bias=True, padding=0)
    layers.append(cl1)
    layers.append(ELU(layers[-1].out_type, inplace=True))

    for i in range(3):
        # every two layers we downsample the feature map
        if i % 2 == 0:
            layers.append(PointwiseAvgPoolAntialiased(layers[-1].out_type, 0.66, stride=2))
        cl = R2Conv(r2, r2, 5, bias=True, padding=0)
        layers.append(cl)
        layers.append(ELU(layers[-1].out_type, inplace=True))

    for i in range(3):
        # every two layers we upsample the feature map
        if i % 2 == 0:
            layers.append(R2Upsampling(layers[-1].out_type, 2, align_corners=True))

        cl = R2Conv(r2, r2, 5, bias=True, padding=0)
        layers.append(cl)
        layers.append(ELU(layers[-1].out_type, inplace=True))

    # finally, map to the output field which will then be visualized
    so2 = SO2(1)

    # A vector field contains two channels transforming according to the frequency-1 irrep of SO(2)
    # (the common 2x2 rotation matrices)
    # the representation needs to be restricted to the group of N discrete rotations considered
    vector_f = FieldType(gc, [so2.irrep(1).restrict(N)])

    # A scalar field contains one channel transforming according to the trivial representation of SO(2)
    # i.e., its values do not change when a rotation is applied
    # the representation needs to be restricted to the group of N discrete rotations considered
    scalar_f = FieldType(gc, [so2.trivial_representation.restrict(N)])

    # build the output field type
    if output == "vector":
        r3 = vector_f
    elif output == "scalar":
        r3 = scalar_f
    elif output == "both":
        # in this case we outputs both a scalar and a vector field
        r3 = scalar_f + vector_f
    else:
        raise ValueError()

    cl2 = R2Conv(layers[-1].out_type, r3, 5, padding=0, bias=False)
    layers.append(cl2)

    # for visualization purpose, apply a non-linearity on the output to restrict the range of values it takes
    if output == "vector":
        layers.append(NormNonLinearity(layers[-1].out_type, "squash", bias=False))
    elif output == "scalar":
        layers.append(SequentialModule(InnerBatchNorm(r3), PointwiseNonLinearity(r3, "p_sigmoid")))
    elif output == "both":
        labels = ["scalar", "vector"]
        nnl = [
            (
                SequentialModule(InnerBatchNorm(scalar_f), PointwiseNonLinearity(scalar_f, "p_sigmoid")),
                "scalar"
            ),
            (
                NormNonLinearity(vector_f, "squash", bias=False),
                "vector"
            ),
        ]
        layers.append(MultipleModule(r3, labels, nnl))
    else:
        raise ValueError()

    model = SequentialModule(*layers)
    return model


if __name__ == "__main__":

    from params import argument_parser
    args = argument_parser()

    # output = "both"
    # output = "vector"
    output = "scalar"
    N = 24


    # read the input image and retrieve the central patch
    IMG_PATH = "./images/pink_flower.jpg"
    IMG_PATH2 = "./images/dog.jpg"
    IMG_PATH3 = './images/hat.jpg'
    image = mpimg.imread(IMG_PATH).transpose((2, 0, 1))
    image2 = mpimg.imread(IMG_PATH2).transpose((2, 0, 1))
    image3 = mpimg.imread(IMG_PATH2).transpose((2, 0, 1))

    px = 314
    D = 1252
    image = image[:, :, 100:452]
    image2 = image2[: ,157:509, 324:676]
    image3 = image3[:, 157:509, 324:676]

    # build a model equivariant to N rotations
    # if args.viz_network=='E2CNN':
    #     model = build_gcnn(N, output).eval()

    model = DnQNet(args, MODEL_CFGS_V3['F'], CLASSIFIER_CFGS['B'])
    model.load_state_dict(torch.load('data/saved_models/dnq_cfg_fb_cifar10.tar'))
    model.feature_mask, model.avg_ratio = model.make_mask(80.5, args)

    vgg = VGG19(CLASSIFIER_CFGS_VGG19['B'])
    vgg.load_state_dict(torch.load('data/saved_models/vgg19_cfg_a_cifar10.tar'))
    # _initialize_weights(vgg)
    vgg = vgg.encoder[:4]


    # build the animation
    if output == "vector":
        animate(model, vgg, image, image2, "animation_vector.mp4", draw_vector_field, R=72, S=129, args=args)
    elif output == "scalar":
        animate(model, vgg, image, image2, "animation_scalar.mp4", draw_scalar_field, R=72, S=161, args=args)
    elif output == "both":
        animate(model, vgg, image, image2, "animation_mixed.mp4", draw_mixed_field, R=72, S=161, args=args)
    else:
        raise ValueError()
