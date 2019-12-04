import sys
sys.path.insert(0, 'Poem-Norm/')
sys.path.insert(0, 'fair_classification/')
import DatasetReader
import Skylines
import Logger
import itertools
import numpy
# from random import seed
# SEED = 50000
# seed(SEED) # set the random seed so that the random permutations can be reproduced again
# numpy.random.seed(SEED)

if __name__ == '__main__':
    name = "data/adult-cleaned.dat"
    if len(sys.argv) > 1:
        name = str(sys.argv[1])

    dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
    dataset.loadDataset(filename = name)

    SCORES = {}
    CLIPPED_DIAGNOSTIC = {}
    UNCLIPPED_DIAGNOSTIC = {}
    ESTIMATORS = ['SelfNormal']
    VAR = ["SVP"]
    APPROACHES = list(itertools.product(ESTIMATORS, VAR))

    for approach in APPROACHES:
        strApproach = str(approach)
        SCORES[strApproach] = []

    for run in range(10):
        print("************************RUN ", run)

        streamer = Logger.DataStream(dataset = dataset, verbose = False)
        features, labels = streamer.generateStream(subsampleFrac = 0.1, replayCount = 1)

        subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
        subsampled_dataset.trainFeatures = features
        subsampled_dataset.trainLabels = labels
        logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False)
        test_score = logger.crf.test()
        print("logger performance: ", test_score)

        replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)

        features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 1)
        replayed_dataset.trainFeatures = features
        replayed_dataset.trainLabels = labels

        sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)
        replayed_dataset.trainFeatures = replayed_dataset.trainFeatures[:, :-1]

        bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = False)

        replayed_dataset.freeAuxiliaryMatrices()
        del replayed_dataset

        bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
        bandit_dataset.createTrainValidateSplit(validateFrac = 0.2)

        numFeatures = numpy.shape(features)[1]-1
        numLabels = numpy.shape(labels)[1]
        coef = numpy.zeros((numFeatures, numLabels), dtype=numpy.longdouble)
        for i in range(numLabels):
            if logger.crf.labeler[i] is not None:
                coef[:, i] = logger.crf.labeler[i].coef_[:, :-1]

        for approach in APPROACHES:
            strApproach = str(approach)
            minVar = 0
            maxVar = -1
            if approach[1] == "SVP":
                minVar = -6
                maxVar = 0
            currDataset = bandit_dataset

            print("MultiLabelExperiment 1: APPROACH ", approach)
            sys.stdout.flush()
            model = Skylines.PRMWrapper(currDataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = minVar, maxV = maxVar,
                                    minClip = 0, maxClip = 0, estimator_type = approach[0], verbose = False, parallel=None, smartStart=coef)
            model.calibrateHyperParams()
            retTime, retClippedDiagnostic, retUnclippedDiagnostic = model.validate()
            # evaluation
            SCORES[strApproach].append(model.test())

            model.freeAuxiliaryMatrices()
            del model

        bandit_dataset.freeAuxiliaryMatrices()
        del bandit_dataset

    for approach in APPROACHES:
        print("Approach: ", approach)
        print("Average: ", numpy.mean(SCORES[str(approach)]))
        print("STD: ", numpy.std(SCORES[str(approach)]))
    dataset.freeAuxiliaryMatrices()
    del dataset








