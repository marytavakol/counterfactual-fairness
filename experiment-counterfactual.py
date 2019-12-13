import sys
sys.path.insert(0, 'Poem-Norm/')
sys.path.insert(0, 'fair_classification/')
import DatasetReader
import Skylines
import Logger
import itertools
import numpy


if __name__ == '__main__':
    name = "data/adult-cleaned.dat"
    if len(sys.argv) > 1:
        name = str(sys.argv[1])

    dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
    dataset.loadDataset(filename = name)

    AUCs = []
    AUC_std = []
    ACCs = []
    ACC_std = []
    Prule = []
    Prule_std = []

    #replay_count = 1
    k = 0.8
    frac = 0.2
    #for frac in numpy.arange(0.1, 1.1, 0.1):
    #for k in numpy.arange(0.2, 2.1, 0.2):
    for replay_count in range(1,2):

        print("----------------------ReplayCount: ", replay_count)
        #print("----------------------k: ", k)
        #print("----------------------frac: ", frac)

        SCORES = {}
        ACCRCY = {}
        PRULES = {}
        ESTIMATORS = ['Vanilla', 'Stochastic', 'SelfNormal']
        VAR = ["ERM", "SVP"]
        APPROACHES = list(itertools.product(ESTIMATORS, VAR))

        for approach in APPROACHES:
            strApproach = str(approach)
            SCORES[strApproach] = []
            ACCRCY[strApproach] = []
            PRULES[strApproach] = []

        n_runs = 10
        for run in range(n_runs):
            #print("************************RUN ", run)

            streamer = Logger.DataStream(dataset = dataset, verbose = False)
            features, labels = streamer.generateStream(subsampleFrac = frac, replayCount = 1)

            subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
            subsampled_dataset.trainFeatures = features
            subsampled_dataset.trainLabels = labels
            logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False, classifier = "crf")
            test_score = logger.crf.test()
            #logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False, classifier = "svm")
            #logger.svm.test()
            #print("logger performance: ", test_score)
            #print("P-ruls is: ", p_rule)

            replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)

            features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = replay_count)
            replayed_dataset.trainFeatures = features
            replayed_dataset.trainLabels = labels

            sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset, k)

            bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = False)

            replayed_dataset.freeAuxiliaryMatrices()
            del replayed_dataset

            bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
            bandit_dataset.createTrainValidateSplit(validateFrac = 0.1)

            bandit_dataset.trainFeatures = bandit_dataset.trainFeatures[:, :-1]

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

                #print("MultiLabelExperiment 1: APPROACH ", approach)
                model = Skylines.PRMWrapper(currDataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = minVar, maxV = maxVar,
                                        minClip = 0, maxClip = 0, estimator_type = approach[0], verbose = False, parallel=None, smartStart=coef)
                model.calibrateHyperParams()

                model.validate()
                # evaluation
                result = model.test()
                #print("Performance: ", result[0])
                #print("P-rule: ", result[1])
                SCORES[strApproach].append(result[0])
                PRULES[strApproach].append(result[1])
                ACCRCY[strApproach].append(result[2])

                model.freeAuxiliaryMatrices()
                del model

            bandit_dataset.freeAuxiliaryMatrices()
            del bandit_dataset

        for approach in APPROACHES:
            #print(SCORES[str(approach)])
            #print(PRULES[str(approach)])
            print("Approach: ", approach)
            auc = numpy.mean(SCORES[str(approach)])
            AUCs.append(auc)
            print("Average AUC: ", auc)
            auc_std = numpy.std(SCORES[str(approach)])/numpy.sqrt(n_runs)
            AUC_std.append(auc_std)
            print("STD-Error AUC: ", auc_std)

            acc = numpy.mean(ACCRCY[str(approach)])
            ACCs.append(acc)
            print("Average ACC: ", acc)
            acc_std = numpy.std(ACCRCY[str(approach)]) / numpy.sqrt(n_runs)
            ACC_std.append(acc_std)
            print("STD-Error ACC: ", acc_std)

            prule = numpy.mean(PRULES[str(approach)])
            Prule.append(prule)
            print("Average P-rule: ", prule)
            prule_std = numpy.std(PRULES[str(approach)])/numpy.sqrt(n_runs)
            Prule_std.append(prule_std)
            print("STD-Error P-rule: ", prule_std)


    print("--------------------------------------------")
    print("auc = ", AUCs)
    print("auc_std = ", AUC_std)
    print("acc = ", ACCs)
    print("acc_std = ", ACC_std)
    print("prule = ", Prule)
    print("prule_std = ", Prule_std)

    dataset.freeAuxiliaryMatrices()
    del dataset








