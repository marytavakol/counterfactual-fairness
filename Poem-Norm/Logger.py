import DatasetReader
import math
import numpy
import numpy.random
import scipy.sparse
import Skylines
import sys
import utils as ut

class DataStream:
    def __init__(self, dataset, verbose):
        self.verbose = verbose

        self.originalFeatures = dataset.trainFeatures.copy()
        self.originalLabels = dataset.trainLabels.copy()
        
        numSamples = numpy.shape(self.originalFeatures)[0]
        permute = numpy.random.permutation(numSamples)
        self.permutedFeatures = self.originalFeatures[permute, :]
        self.permutedLabels = self.originalLabels[permute, :]

        if self.verbose:
            print("DataStream: [Message] Initialized with permutation over [n_samples]: ", numSamples)
            sys.stdout.flush()

    def generateStream(self, subsampleFrac, replayCount):
        numSamples = numpy.shape(self.permutedFeatures)[0]
        numSubsamples = math.ceil(subsampleFrac*numSamples)
        subsampleFeatures = self.permutedFeatures[0:numSubsamples,:]
        subsampleLabels = self.permutedLabels[0:numSubsamples,:]
        if self.verbose:
            print("DataStream: [Message] Selected subsamples [n_subsamples, n_samples]: ", numSubsamples, numSamples)
            sys.stdout.flush()

        if replayCount <= 1: 
            return subsampleFeatures, subsampleLabels
        else:
            replicator = numpy.ones((replayCount, 1))
            repeatedFeatures = scipy.sparse.kron(replicator, subsampleFeatures, format='csr')
            repeatedLabels = numpy.kron(replicator, subsampleLabels)

            if self.verbose:
                print("DataStream: [Message] Replay samples ", numpy.shape(repeatedFeatures)[0])
                sys.stdout.flush()
            return repeatedFeatures, repeatedLabels

    def freeAuxiliaryMatrices(self):
        del self.originalFeatures
        del self.originalLabels
        del self.permutedFeatures
        del self.permutedLabels
        
        if self.verbose:
            print("Datastream: [Message] Freed matrices")
            sys.stdout.flush()


class Logger:
    def __init__(self, dataset, loggerC, stochasticMultiplier, verbose):
        self.verbose = verbose
        crf = Skylines.CRF(dataset = dataset, tol = 1e-5, minC = loggerC, maxC = loggerC, verbose = self.verbose, parallel = True)
        crf.Name = "LoggerCRF"
        crf.validate()
        if not(stochasticMultiplier == 1):
            for i in range(len(crf.labeler)):
                if crf.labeler[i] is not None:
                    crf.labeler[i].coef_ = stochasticMultiplier * crf.labeler[i].coef_

        self.crf = crf
        if self.verbose:
            print("Logger: [Message] Trained logger crf. Weight-scale: ", stochasticMultiplier)
            sys.stdout.flush()

    def freeAuxiliaryMatrices(self):
        del self.crf




    def generateLog(self, dataset):
        numSamples, numFeatures = numpy.shape(dataset.trainFeatures)
        numLabels = numpy.shape(dataset.trainLabels)[1]

        sampledLabels = dataset.trainLabels.copy()
        logpropensity = numpy.zeros(numSamples, dtype = numpy.longdouble)
        for i in range(numLabels):
            if self.crf.labeler[i] is not None:
                regressor = self.crf.labeler[i]
                predictedProbabilities = regressor.predict_log_proba(dataset.trainFeatures)

                probSampledLabel = numpy.zeros(numSamples, dtype=numpy.longdouble)
                probSampledLabel[sampledLabels[:, 0] > 0] = predictedProbabilities[sampledLabels[:, 0] > 0, 1]
                probSampledLabel[sampledLabels[:, 0] < 1] = predictedProbabilities[sampledLabels[:, 0] < 1, 0]
                logpropensity = logpropensity + probSampledLabel

        x_control = dataset.trainFeatures[:, -1].todense()
        control_num = ut.compute_imbalance(x_control, dataset.trainLabels)
        actualProb = numpy.exp(predictedProbabilities)
        prot_neg = numpy.where((x_control == 0.0) & (dataset.trainLabels == 0.0))[0]
        non_prot_pos = numpy.where((x_control == 1.0) & (dataset.trainLabels == 1.0))[0]
        prot_low_prob = numpy.argsort(actualProb[prot_neg, 0])[:control_num]
        non_prot_low_prob = numpy.argsort(actualProb[non_prot_pos, 1])[:control_num]

        sampledLoss = numpy.zeros(numSamples) - 1

        prot_ind = prot_neg[prot_low_prob]
        non_prot_ind = non_prot_pos[non_prot_low_prob]

        sampledLoss[prot_ind] = 0
        sampledLoss[non_prot_ind] = 0


        # rand = numpy.random.rand(numSamples, 1)
        # non_prot_ind = numpy.where((x_control == 1.0) & (dataset.trainLabels == 1.0) & (rand < non_prot_prob))[0]
        # prot_ind = numpy.where((x_control == 0.0) & (dataset.trainLabels == 0.0) & (rand < prot_prob))[0]
        # sampledLoss[prot_ind] = 0
        # sampledLoss[non_prot_ind] = 0

        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = numpy.longdouble)
            print("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, numpy.shape(sampledLabels)[0])
            print("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()
        return sampledLabels, logpropensity, sampledLoss

