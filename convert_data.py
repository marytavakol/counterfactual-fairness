import DatasetReader
import Skylines
import Logger
import PRM
import numpy
import sys
import PDTTest
import itertools
import operator

if __name__ == '__main__':
    name = "adult.dat"
    if len(sys.argv) > 1:
        exptNum = int(sys.argv[1])


    if exptNum == 1:
        for name in ['scene', 'yeast', 'rcv1_topics', 'tmc2007']:
            dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = True)
            if name == 'rcv1_topics':
                dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])
            else:
                dataset.loadDataset(corpusName = name)

            SCORES = {}
            TIMES = {}
            CLIPPED_DIAGNOSTIC = {}
            UNCLIPPED_DIAGNOSTIC = {}
            SKYLINES = ["SVM", "CRF"]
            ESTIMATORS = ['Vanilla', 'SelfNormal']
            VAR = ["ERM", "SVP"]
            WARMSTART = [False]
            TRANSLATELOSS = [True, False]
            APPROACHES = list(itertools.product(ESTIMATORS, VAR, WARMSTART, TRANSLATELOSS))
            for approach in SKYLINES:
                TIMES[approach] = []
                SCORES[approach] = []
                if approach == "CRF":
                    SCORES[approach+"_expected"] = []
            SCORES['Logger'] = []
            SCORES['Logger_map'] = []
            for approach in APPROACHES:
                strApproach = str(approach)
                TIMES[strApproach] = []
                CLIPPED_DIAGNOSTIC[strApproach] = []
                UNCLIPPED_DIAGNOSTIC[strApproach] = []
                SCORES[strApproach] = []
                SCORES[strApproach+"_map"] = []

            for run in xrange(10):
                print("************************RUN ", run)
                
                supervised_dataset = DatasetReader.SupervisedDataset(dataset = dataset, verbose = True)
                supervised_dataset.createTrainValidateSplit(validateFrac = 0.25)

                for approach in SKYLINES:
                    model = None
                    if approach == "SVM":
                        model = Skylines.SVM(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = None)
                    elif approach == "CRF":
                        model = Skylines.CRF(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = pool)
                    else:
                        print("ERROR:", approach, " not found.")
                        sys.stdout.flush()
                        sys.exit()
                    TIMES[approach].append(model.validate()[0])
                    SCORES[approach].append(model.test())
                    if approach == "CRF":
                        SCORES[approach+"_expected"].append(model.expectedTestLoss())

                supervised_dataset.freeAuxiliaryMatrices()
                del supervised_dataset
                
                streamer = Logger.DataStream(dataset = dataset, verbose = True)
                features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

                subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)
                subsampled_dataset.trainFeatures = features
                subsampled_dataset.trainLabels = labels
                logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = True)
                SCORES['Logger_map'].append(logger.crf.test())
                SCORES['Logger'].append(logger.crf.expectedTestLoss())

                replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)

                features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
                replayed_dataset.trainFeatures = features
                replayed_dataset.trainLabels = labels

                sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)

                bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = True)

                replayed_dataset.freeAuxiliaryMatrices()
                del replayed_dataset

                bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
                bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)

                numFeatures = numpy.shape(features)[1]
                numLabels = numpy.shape(labels)[1] 
                coef = numpy.zeros((numFeatures, numLabels), dtype = numpy.longdouble)
                for i in xrange(numLabels):
                    if logger.crf.labeler[i] is not None:
                        coef[:,i] = logger.crf.labeler[i].coef_

                logger.freeAuxiliaryMatrices()  
                del logger

                bandit_dataset_scaled = DatasetReader.BanditDataset(dataset = bandit_dataset, verbose = True)
                bandit_dataset_scaled.validateFeatures = bandit_dataset.validateFeatures
                bandit_dataset_scaled.validateLabels = bandit_dataset.validateLabels
                bandit_dataset_scaled.trainSampledLabels = bandit_dataset.trainSampledLabels
                bandit_dataset_scaled.validateSampledLabels = bandit_dataset.validateSampledLabels
                bandit_dataset_scaled.trainSampledLogPropensity = bandit_dataset.trainSampledLogPropensity
                bandit_dataset_scaled.validateSampledLogPropensity = bandit_dataset.validateSampledLogPropensity
                bandit_dataset_scaled.trainSampledLoss = bandit_dataset.trainSampledLoss.copy()
                bandit_dataset_scaled.trainSampledLoss += numLabels
                bandit_dataset_scaled.validateSampledLoss = bandit_dataset.validateSampledLoss.copy()
                bandit_dataset_scaled.validateSampledLoss += numLabels

                for approach in APPROACHES:
                    strApproach = str(approach)
                    startW = None
                    if approach[2]:
                        startW = coef
                    minVar = 0
                    maxVar = -1
                    if approach[1] == "SVP":
                        minVar = -6
                        maxVar = 0
                    currDataset = None
                    if approach[3]:
                        currDataset = bandit_dataset
                    else:
                        currDataset = bandit_dataset_scaled
                    print("MultiLabelExperiment 1: APPROACH ", approach)
                    sys.stdout.flush()
                    model = Skylines.PRMWrapper(currDataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = minVar, maxV = maxVar,
                                            minClip = 0, maxClip = 0, estimator_type = approach[0], verbose = True,
                                            parallel = pool, smartStart = startW)
                    model.calibrateHyperParams()
                    retTime, retClippedDiagnostic, retUnclippedDiagnostic = model.validate()
                    TIMES[strApproach].append(retTime)
                    CLIPPED_DIAGNOSTIC[strApproach].append(retClippedDiagnostic)
                    UNCLIPPED_DIAGNOSTIC[strApproach].append(retUnclippedDiagnostic)
                    SCORES[strApproach+"_map"].append(model.test())
                    SCORES[strApproach].append(model.expectedTestLoss())

                    model.freeAuxiliaryMatrices()
                    del model

                bandit_dataset.freeAuxiliaryMatrices()  
                del bandit_dataset

                bandit_dataset_scaled.freeAuxiliaryMatrices()  
                del bandit_dataset_scaled
 
            dataset.freeAuxiliaryMatrices()  
            del dataset
            
            print("************** RESULTS FOR ", name)
            print(SCORES)
            print(TIMES)
            print(CLIPPED_DIAGNOSTIC)
            print(UNCLIPPED_DIAGNOSTIC)
            print("*******************************")
            sys.stdout.flush()


            print("############### SUMMARY", name)
            print("TIMES")
            sys.stdout.flush()
            for key in TIMES:
                obs = PDTTest.ExperimentResult(TIMES[key], verbose = False)
                print key, obs.reportMean()
            print "CLIPPED DIAGNOSTIC"
            sys.stdout.flush()
            for key in CLIPPED_DIAGNOSTIC:
                obs = PDTTest.ExperimentResult(CLIPPED_DIAGNOSTIC[key], verbose = False)
                print key, obs.reportMean()
            print "UNCLIPPED DIAGNOSTIC"
            sys.stdout.flush()
            for key in UNCLIPPED_DIAGNOSTIC:
                obs = PDTTest.ExperimentResult(UNCLIPPED_DIAGNOSTIC[key], verbose = False)
                print key, obs.reportMean()

            SUMMARY_SCORES = {}
            MEAN_SCORES = {}
            print "SCORES"
            sys.stdout.flush()
            for key in SCORES:
                obs =  PDTTest.ExperimentResult(SCORES[key], verbose = False)
                MEAN_SCORES[key] = obs.reportMean()
                SUMMARY_SCORES[key] = obs

            sorted_x = sorted(MEAN_SCORES.items(), key = operator.itemgetter(1))
            for tup in sorted_x:
                print tup

            for key1 in SUMMARY_SCORES:
                for key2 in SUMMARY_SCORES:
                    if key1 == key2:
                        continue
                    obs1 = SUMMARY_SCORES[key1]
                    obs2 = SUMMARY_SCORES[key2]
                    if obs1.testDifference(obs2):
                        print key1, ">", key2

            sys.stdout.flush()


