import caffe
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import ifilter
import argparse
import os

caffe.set_mode_cpu()

parser = argparse.ArgumentParser()
parser.add_argument('--preTrainedRoot', type=str, required=True, dest='preTrainedRoot')
parser.add_argument('--imageToFool', type=str, required=True, dest='imageToFool')
parser.add_argument('--intendedResult', type=int, required=True, dest='intendedResult')
args = vars(parser.parse_args())

preTrainedRoot, imageToFool, intendedOutcome = args['preTrainedRoot'], args['imageToFool'], args['intendedResult']
maxIterations, outputPath  = 500, 'outputImages'

# -----------------------------------------------
# -----------------------------------------------
def loadGoogLeNet():
    batchSize = 1
    googleNet = caffe.Net(preTrainedRoot + "deploy.prototxt", preTrainedRoot + "bvlc_googlenet.caffemodel", caffe.TEST)
    dataLayerShape = (batchSize, ) + googleNet.blobs['data'].data.shape[1:]
    googleNet.blobs['data'].reshape(*dataLayerShape)
    googleNet.blobs['prob'].reshape(batchSize, )
    googleNet.reshape()
    return googleNet


def makeCaffeTransformer():
    transformer = caffe.io.Transformer({'data': googLeNet.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    return transformer


def displayImage(img, imgName):
    topPredictedClasses = forwardPassPredictions(img)[:2]
    img = transformImage(img)
    plt.imshow(img)
    plt.title(topPredictedClasses)
    plt.savefig(imgName)
    plt.close()


def transformImage(img):
    return caffeTransformer.deprocess('data', img)


def displayPerturbation(inputImage, finalImage, diffName):
    imageDiff = transformImage(finalImage - inputImage)
    scaleFactor = float(imageDiff.max() - imageDiff.min())
    reScaledImageDiff = (imageDiff - imageDiff.min()) / scaleFactor
    print 'scaleFactor for difference plot is = %f' % scaleFactor
    plt.imshow(reScaledImageDiff)
    plt.savefig(diffName)


def simplifyLabel(label):
    return label.split(',')[0]


def joinPredictLabel(predictedClasses):
    labeledPredictions = [(pred[1], imagenetLabels.loc[pred[0]][0]) for pred in predictedClasses]
    return map(lambda x: (round(100 * x[0], 1), simplifyLabel(x[1])), labeledPredictions)


def forwardPassPredictions(img):
    googLeNet.blobs['data'].data[...] = img
    outputProb = googLeNet.forward()
    predictedClasses = sorted(enumerate(outputProb['prob'][0]), key=itemgetter(1), reverse=True)
    return joinPredictLabel(predictedClasses)


def modifyImage(img):
    existingPredictions = forwardPassPredictions(img)               # SIDE-EFFECT: updates prob blob because of forward pass
    lossGradient = googLeNet.backward(prob=desiredProb)['data']     # calculates loss and associated gradients
    return img + np.sign(lossGradient) * 0.01


def deformationGenerator(deformationFunction, currentImage):
    for _ in range(maxIterations):
        newImage = deformationFunction(currentImage)
        yield newImage
        currentImage = newImage


def captureNetworkWeights():
    return [(googLeNet.params[params][0].data, googLeNet.params[params][1].data) for params in googLeNet.params]


# Also logs some information into outer list
def constraintSatisfaction(x):
    prediction = forwardPassPredictions(x)
    print prediction[:8]
    return prediction[0][1] == simplifyLabel(intendedLabel) and prediction[0][0] > 80

# -----------------------------------------------
# -----------------------------------------------

imagenetLabels = pd.read_table("imageNet.labels", header=None)

googLeNet = loadGoogLeNet()
caffeTransformer = makeCaffeTransformer()

inputImage = caffeTransformer.preprocess('data', caffe.io.load_image(imageToFool))

intendedLabel = imagenetLabels.loc[intendedOutcome][0]
print "Intended outcome is: %s" % simplifyLabel(intendedLabel)
desiredProb = np.zeros_like(googLeNet.blobs['prob'].data)
desiredProb[0][intendedOutcome] = 1  # set to 1 for the intended outcome

processDeformation = deformationGenerator(modifyImage, inputImage)

initialWeights = captureNetworkWeights()
finalImage = next(ifilter(constraintSatisfaction, processDeformation), 'neoir')
finalWeights = captureNetworkWeights()

finalImage = finalImage[0]

# make sure that the network did not change at all
assert all([np.array_equal(weights[0][0], weights[1][0]) for weights in zip(initialWeights, finalWeights)])
assert all([np.array_equal(weights[0][1], weights[1][1]) for weights in zip(initialWeights, finalWeights)])

picName = os.path.basename(imageToFool).split('.')[0]
os.makedirs(os.path.join(outputPath, picName))

displayPerturbation(inputImage, finalImage, os.path.join(outputPath, picName, picName + '.diff.jpg'))
displayImage(inputImage, os.path.join(outputPath, picName, picName + '.input.jpg'))
displayImage(finalImage, os.path.join(outputPath, picName, picName + '.fooled.jpg'))
