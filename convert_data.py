import sys
sys.path.insert(0, 'Poem-Norm/')
sys.path.insert(0, 'fair_classification/')
import DatasetReader
import Skylines
import Logger
import itertools

if __name__ == '__main__':
    name = "data/adult-cleaned.dat"
    if len(sys.argv) > 1:
        name = str(sys.argv[1])

    dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
    dataset.loadDataset(filename = name)

    SCORES = {}
    TIMES = {}
    CLIPPED_DIAGNOSTIC = {}
    UNCLIPPED_DIAGNOSTIC = {}
    ESTIMATORS = ['SelfNormal']
    VAR = ["ERM", "SVP"]
    APPROACHES = list(itertools.product(ESTIMATORS, VAR))

    for approach in APPROACHES:
        strApproach = str(approach)
        TIMES[strApproach] = []
        CLIPPED_DIAGNOSTIC[strApproach] = []
        UNCLIPPED_DIAGNOSTIC[strApproach] = []
        SCORES[strApproach] = []
        SCORES[strApproach+"_map"] = []

    for run in range(1):
        print("************************RUN ", run)

        streamer = Logger.DataStream(dataset = dataset, verbose = False)
        features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

        subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
        subsampled_dataset.trainFeatures = features
        subsampled_dataset.trainLabels = labels
        logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False)
        logger.crf.test()

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
                                    minClip = 0, maxClip = 0, estimator_type = approach[0], verbose = True, parallel=None, smartStart=None)
            model.calibrateHyperParams()
            retTime, retClippedDiagnostic, retUnclippedDiagnostic = model.validate()

            # evaluation
            SCORES[strApproach+"_map"].append(model.test())

            model.freeAuxiliaryMatrices()
            del model

        bandit_dataset.freeAuxiliaryMatrices()
        del bandit_dataset

    dataset.freeAuxiliaryMatrices()
    del dataset








