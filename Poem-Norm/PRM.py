import itertools
import numpy
import numpy.random
import scipy.optimize
import scipy.special
import sklearn.base
import sys
import utils as ut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class PRMEstimator(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, n_iter, tol, l2reg, varpenalty, clip, verbose):
        self.n_iter = n_iter
        self.tol = tol
        self.l2reg = l2reg
        self.varpenalty = varpenalty
        self.clip = clip
        self.verbose = verbose


class MultiLabelEstimator(PRMEstimator):
    def __init__(self, n_iter, tol, l2reg, varpenalty, clip, verbose):
        PRMEstimator.__init__(self, n_iter, tol, l2reg, varpenalty, clip, verbose)
        self.y = None
        self.ySign = None
        self.logpropensity = None
        self.sampledLoss = None
        self.X = None
        self.Xtranspose = None

    def ObjectiveOnly(self, w):
        retVal = self.Objective(w)
        return retVal[0]

    def GradientOnly(self, w):
        retVal = self.Objective(w)
        return retVal[1]

    def setAuxiliaryMatrices(self, X, y, logpropensity, sampledLoss):
        self.y = y
        self.ySign = 2*y - 1
        self.logpropensity = logpropensity
        self.sampledLoss = sampledLoss
        self.X = X
        self.Xtranspose = X.transpose()
        numLabels = numpy.shape(self.y)[1]

        if self.verbose:
            print("MultiLabelEstimator: [Message] Loaded matrices")
            sys.stdout.flush()

    def freeAuxiliaryMatrices(self):
        if self.y is not None:
            del self.y
            del self.ySign
            del self.logpropensity
            del self.sampledLoss
            del self.X
            del self.Xtranspose
            if self.verbose:
                print("MultiLabelEstimator: [Message] Freed matrices")
                sys.stdout.flush()

    def checkGradient(self):
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]

        for i in range(10):
            startW = 10*numpy.random.randn(numFeatures*numLabels)
            print("MultiLabelEstimator: [Message] CheckGradient ", scipy.optimize.check_grad(self.ObjectiveOnly, self.GradientOnly, startW))
            sys.stdout.flush()

    def fit(self, start_point):
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]

        ops = {'maxiter': self.n_iter, 'disp': self.verbose, 'gtol': self.tol,\
            'ftol': self.tol, 'maxcor': 50}
        
        if start_point is not None:
            startW = numpy.reshape(start_point, numFeatures*numLabels)
        else:
            startW = numpy.zeros((numFeatures, numLabels), dtype = numpy.longdouble)
            startW = numpy.reshape(startW, numFeatures*numLabels)

        Result = scipy.optimize.minimize(fun = self.Objective, x0 = startW.astype(numpy.float_),
                    method = 'L-BFGS-B', jac = True, tol = self.tol, options = ops)

        if self.verbose:
            print("MultiLabelEstimator: [Message] Finished optimization ", Result['message'])
            sys.stdout.flush()

        self.coef_ = Result['x']
        """
        if start_point is None:
            bestVal = Result['fun']

            for restart in xrange(5):
                if self.verbose:
                    print "MultiLabelEstimator: [Message] Starting random restarts ", restart
                    sys.stdout.flush()

                startW = numpy.random.rand(numFeatures, numLabels)
                startW = numpy.reshape(startW, numFeatures*numLabels)

                Result = scipy.optimize.minimize(fun = self.Objective, x0 = startW.astype(numpy.float_),
                            method = 'L-BFGS-B', jac = True, tol = self.tol, options = ops)
                val = Result['fun']
                if val < bestVal:
                    bestVal = val
                    self.coef_ = Result['x']
        """
        self.coef_ = numpy.reshape(self.coef_, (numFeatures, numLabels))
        numSamples = numpy.shape(self.logpropensity)[0]
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]

        WX = self.X.dot(self.coef_)
        YWX = numpy.multiply(WX, self.ySign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)

        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - self.logpropensity
        unclippedWeights = numpy.exp(logImportanceSampleWeights)
        unclippedWeights[zeroMask] = 0.0
        unclippedDiagnostic = unclippedWeights.mean(dtype = numpy.longdouble)

        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("MultiLabelEstimator: [Debug] Fit Diagnostic C=", mask.sum())
            sys.stdout.flush()

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0
        clippedDiagnostic = ImportanceSampleWeights.mean(dtype = numpy.longdouble)
        return clippedDiagnostic, unclippedDiagnostic
   
    def predict(self, X):
        numSamples = numpy.shape(X)[0]

        WX = X.dot(self.coef_)
        predictions = (WX >= 0).astype(int)
        return predictions

    def computeExpectedLoss(self, X, Y):
        numSamples = numpy.shape(X)[0]
        numLabels = numpy.shape(Y)[1]

        WX = X.dot(self.coef_)
        YSign = 1 - 2*Y
        YWX = numpy.multiply(WX, YSign)
        LossPerInstanceLabel = scipy.special.expit(YWX)
        marginal = LossPerInstanceLabel.sum(dtype = numpy.longdouble)
        return marginal / (numSamples * numLabels)

    def computeCfactHammingLoss(self, X, Y, delta, logpropensity):
        numSamples = numpy.shape(X)[0]
        numLabels = numpy.shape(Y)[1]
        translated_delta = delta

        WX = X.dot(self.coef_)
        YSign = 2*Y - 1
        YWX = numpy.multiply(WX, YSign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)
        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)
        logImportanceSampleWeights = logProbPerInstance - logpropensity

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(translated_delta, ImportanceSampleWeights)
        sumWeightedLoss = WeightedLossPerInstance.sum(dtype = numpy.longdouble)
        sumWeights = ImportanceSampleWeights.sum(dtype = numpy.longdouble)
        meanWeightedLoss = sumWeightedLoss / sumWeights

        return 1.0 + (meanWeightedLoss / numLabels)

    def computeValidationLoss(self, X, Y):

        predictedLabels = self.predict(X[:, :-1])
        #acc = accuracy_score(Y, predictedLabels)
        #f1 = f1_score(Y, predictedLabels)
        auc = roc_auc_score(Y, predictedLabels)
        #print("Accuracy: ", acc)
        #print("F1-measure: ", f1)
        #print("AUC: ", auc)
        pval = ut.compute_p_rule(X[:, -1].todense(), predictedLabels)
        #print("P-rule: ", pval)

        #error = 2 - auc - pval
        error = 1 - auc
        #print("Error", error)

        return error

class VanillaISEstimator(MultiLabelEstimator):
    def Objective(self, w):
        w = w.astype(numpy.longdouble)
        numSamples = numpy.shape(self.logpropensity)[0]
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]
        currW = numpy.reshape(w, (numFeatures, numLabels))

        Obj = 0
        L2Obj = 0
        VarObj = 0
        Grad = None
        L2Grad = numpy.zeros(numpy.shape(w), dtype = numpy.longdouble)
        VarGrad = numpy.zeros(numpy.shape(currW), dtype = numpy.longdouble)

        WX = self.X.dot(currW)
        YWX = numpy.multiply(WX, self.ySign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)

        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - self.logpropensity
        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("VanillaISEstimator: [Debug] C=", mask.sum())
            sys.stdout.flush()

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(self.sampledLoss, ImportanceSampleWeights)
        
        #This is \hat{Err}(w)
        meanWeightedLoss = WeightedLossPerInstance.mean(dtype = numpy.longdouble)

        #Computing Grad \hat{Err}(w)
        PartitionPerInstanceLabel = scipy.special.expit(WX)
        LabelPartitionPerInstanceLabel = self.y - PartitionPerInstanceLabel
        LabelPartitionPerInstanceLabel[mask, :] = 0

        LossLabelPartitionPerInstanceLabel = numpy.multiply(LabelPartitionPerInstanceLabel,
            WeightedLossPerInstance[:,None])

        #This is Grad \hat{Err}(w)
        g = self.Xtranspose.dot(LossLabelPartitionPerInstanceLabel) / numSamples
        
        if self.varpenalty > 0:
            diffWtLossPerInstance = WeightedLossPerInstance - meanWeightedLoss
            sqrtVar = scipy.linalg.norm(diffWtLossPerInstance)
            sqrtN = numpy.sqrt(numSamples * (numSamples-1))

            if sqrtVar > 0:
                VarObj = sqrtVar / sqrtN

                WeightedGradient = numpy.multiply(LossLabelPartitionPerInstanceLabel,
                    WeightedLossPerInstance[:,None])
                VarGrad = self.Xtranspose.dot(WeightedGradient)
                VarGrad -= meanWeightedLoss*numSamples * g
                VarGrad /= sqrtN * sqrtVar

        if self.l2reg > 0:
            wNorm = numpy.dot(w,w)
            L2Obj = wNorm
        
            L2Grad = 2 * w

        Obj = meanWeightedLoss + self.l2reg * L2Obj + self.varpenalty * VarObj
        Grad = numpy.reshape(g,numFeatures*numLabels)  + self.l2reg * L2Grad +\
            self.varpenalty * numpy.reshape(VarGrad, numFeatures*numLabels)

        if self.verbose:
            print(".",)
            sys.stdout.flush()
       
        return (Obj, Grad.astype(numpy.float_))


class SelfNormalEstimator(MultiLabelEstimator):
    def Objective(self, w):
        w = w.astype(numpy.longdouble)
        numSamples = numpy.shape(self.logpropensity)[0]
        numLabels = numpy.shape(self.y)[1]
        numFeatures = numpy.shape(self.X)[1]
        currW = numpy.reshape(w, (numFeatures, numLabels))

        Obj = 0
        L2Obj = 0
        VarObj = 0
        Grad = None
        L2Grad = numpy.zeros(numpy.shape(w), dtype = numpy.longdouble)
        VarGrad = numpy.zeros(numpy.shape(currW), dtype = numpy.longdouble)

        WX = self.X.dot(currW)
        YWX = numpy.multiply(WX, self.ySign)
        ProbPerInstanceLabel = scipy.special.expit(YWX)

        zeroMask = ProbPerInstanceLabel <= 0
        ProbPerInstanceLabel[zeroMask] = 1.0
        zeroEntries = zeroMask.sum(axis = 1, dtype = numpy.int)
        zeroMask = zeroEntries > 0

        logProbPerInstanceLabel = numpy.log(ProbPerInstanceLabel)
        logProbPerInstance = numpy.sum(logProbPerInstanceLabel, axis = 1, dtype = numpy.longdouble)

        logImportanceSampleWeights = logProbPerInstance - self.logpropensity
        mask = numpy.logical_and(logImportanceSampleWeights >= self.clip, numpy.logical_not(zeroMask))
        logImportanceSampleWeights[mask] = self.clip

        if self.verbose:
            print("VanillaISEstimator: [Debug] C=", mask.sum())
            sys.stdout.flush()

        ImportanceSampleWeights = numpy.exp(logImportanceSampleWeights)
        ImportanceSampleWeights[zeroMask] = 0.0

        WeightedLossPerInstance = numpy.multiply(self.sampledLoss, ImportanceSampleWeights)
        
        totalWeightedLoss = WeightedLossPerInstance.sum(dtype = numpy.longdouble)
        totalWeights = ImportanceSampleWeights.sum(dtype = numpy.longdouble)

        if totalWeights <= 0.0:
            estimatedLoss = 0.0
            VarObj = 0.0
            g = numpy.zeros(numpy.shape(currW), dtype = numpy.longdouble)
        else:
            #This is \hat{Err}(w)
            estimatedLoss = totalWeightedLoss * 1.0 / totalWeights

            PartitionPerInstanceLabel = scipy.special.expit(WX)
            LabelPartitionPerInstanceLabel = self.y - PartitionPerInstanceLabel
            LabelPartitionPerInstanceLabel[mask, :] = 0

            LossLabelPartitionPerInstanceLabel = numpy.multiply(LabelPartitionPerInstanceLabel,
                WeightedLossPerInstance[:,None])
            GradObjNumerator = self.Xtranspose.dot(LossLabelPartitionPerInstanceLabel) / totalWeights

            WeightLabelPartitionPerInstanceLabel = numpy.multiply(LabelPartitionPerInstanceLabel,
                ImportanceSampleWeights[:,None])
            GradObjDenominator = self.Xtranspose.dot(WeightLabelPartitionPerInstanceLabel)
            WeightedGradObjDenominator = GradObjDenominator * estimatedLoss / totalWeights

            #This is Grad \hat{Err}(w)
            g = GradObjNumerator - WeightedGradObjDenominator

            if self.varpenalty > 0:
                normalizedWeights = ImportanceSampleWeights / totalWeights
                diffLossPerInstance = self.sampledLoss - estimatedLoss

                VarPerInstance = numpy.multiply(normalizedWeights, diffLossPerInstance)
                sqrtVar = scipy.linalg.norm(VarPerInstance)
                sqrtN = numpy.sqrt(numSamples)

                if sqrtVar > 0:
                    VarObj = sqrtVar / sqrtN

                    WtVarPerInstance = numpy.multiply(VarPerInstance, normalizedWeights)
                    FirstTerm = -WtVarPerInstance.sum(dtype = numpy.longdouble) * g
                    FirstTerm /= sqrtN * sqrtVar

                    LossVarPerInstance = numpy.multiply(VarPerInstance, diffLossPerInstance) / totalWeights
                    WtLossVarPerInstance = numpy.multiply(WeightLabelPartitionPerInstanceLabel,
                        LossVarPerInstance[:, None])
                    SecondTerm = self.Xtranspose.dot(WtLossVarPerInstance)
                    SecondTerm /= sqrtN * sqrtVar

                    ThirdTerm = -VarObj*GradObjDenominator / totalWeights

                    VarGrad = FirstTerm + SecondTerm + ThirdTerm

        if self.l2reg > 0:
            wNorm = numpy.dot(w,w)
            L2Obj = wNorm

            L2Grad = 2 * w

        Obj = estimatedLoss + self.l2reg * L2Obj + self.varpenalty * VarObj
        Grad = numpy.reshape(g,numFeatures*numLabels)  + self.l2reg * L2Grad +\
            self.varpenalty * numpy.reshape(VarGrad, numFeatures*numLabels)

        if self.verbose:
            print("*",)
            sys.stdout.flush()

        return (Obj, Grad.astype(numpy.float_))
