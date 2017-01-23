import caffe
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import ifilter
import argparse

caffe.set_mode_cpu()

parser = argparse.ArgumentParser()
parser.add_argument('--maxIterations', type=int, required=True, dest='maxIterations')
parser.add_argument('--preTrainedRoot', type=str, required=True, dest='preTrainedRoot')
args = vars(parser.parse_args())

maxIterations, preTrainedRoot = args['maxIterations'], args['preTrainedRoot']

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
    topPredictedClasses = simplePredictor(img)[:2]
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
    #return map(lambda x: (round(100 * x[0], 1), x[1].split(',')[0]), labeledPredictions)
    return map(lambda x: (round(100 * x[0], 1), simplifyLabel(x[1])), labeledPredictions)


def simplePredictor(inputImage):
    googLeNet.blobs['data'].data[...] = inputImage
    outputProb = googLeNet.forward()
    predictedClasses = sorted(enumerate(outputProb['prob'][0]), key=itemgetter(1), reverse=True)
    return joinPredictLabel(predictedClasses)


def modifyImage(img):  # Also logs some information into outer list
    backGradient = googLeNet.backward(prob=desiredProb)['data']
    newImage = img + np.sign(backGradient) * 0.01
    allPredictions = simplePredictor(newImage)
    print allPredictions[:10]
    mostLikelyPrediction = allPredictions[0]
    logIntermediatePredictions.append(mostLikelyPrediction)
    return newImage


def deformationGenerator(deformationFunction, currentState):
    for _ in range(maxIterations):
        newState = deformationFunction(currentState)
        yield newState
        currentState = newState


def captureNetworkState():
    blobs = [googLeNet.blobs[blobs].data for blobs in googLeNet.blobs]
    weights = [(googLeNet.params[params][0].data, googLeNet.params[params][1].data) for params in googLeNet.params]
    return blobs, weights


# -----------------------------------------------
# -----------------------------------------------

imagenetLabels = pd.read_table("imageNet.labels", header=None)

googLeNet = loadGoogLeNet()
caffeTransformer = makeCaffeTransformer()

imageName = "pizza.jpg"

nameInput = "inputImages/" + imageName

#inputImage = caffeTransformer.preprocess('data', caffe.io.load_image("Typhoon3.jpg"))
inputImage = caffeTransformer.preprocess('data', caffe.io.load_image(nameInput))
predictedClasses = simplePredictor(inputImage)

# -------------------------------------------
# Choose the intended result to fool the network
# -------------------------------------------

intendedOutcome = 30
intendedLabel = imagenetLabels.loc[intendedOutcome][0]
print "Intended outcome is: %s" % simplifyLabel(intendedLabel)

desiredProb = np.zeros_like(googLeNet.blobs['prob'].data)
desiredProb[0][intendedOutcome] = 1

# -------------------------------------------

logIntermediatePredictions = []
processDeformation = deformationGenerator(modifyImage, inputImage)

initialBlobs, initialWeights = captureNetworkState()
#TODO: create function for constraint statisfaction
finalImage = next(ifilter(lambda x: simplePredictor(x)[0][1] == simplifyLabel(intendedLabel) and simplePredictor(x)[0][0] > 80, processDeformation), 'neoir')
finalBlobs, finalWeights = captureNetworkState()

finalImage = finalImage[0]

assert all(np.array_equal(blob[0], blob[1]) for blob in zip(initialBlobs, finalBlobs))
assert all([np.array_equal(weights[0][0], weights[1][0]) for weights in zip(initialWeights, finalWeights)])
assert all([np.array_equal(weights[0][1], weights[1][1]) for weights in zip(initialWeights, finalWeights)])

displayPerturbation(inputImage, finalImage, 'outputImages/diff.%s' % imageName)
displayImage(inputImage, 'outputImages/input.%s' % imageName)
displayImage(finalImage, 'outputImages/output.%s' % imageName)

'''
#hist, bin_edges = np.histogram(imageDiff.flatten(), bins = 20)
#plt.bar(bin_edges[:-1], hist, width = 1, color='g')
'''
