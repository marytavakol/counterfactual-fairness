import sys
import numpy

class ExperimentResult:
    def __init__(self, resultList, verbose):
        self.verbose = verbose
        self.numSamples = len(resultList)
        if self.numSamples <= 0 and self.verbose:
            print("No observations provided. Aborting.")
            sys.stdout.flush()
            return

        self.results = numpy.array(resultList, dtype = numpy.longdouble)

    def reportMean(self):
        avg = self.results.mean(dtype = numpy.longdouble)
        return avg

    def testDifference(self, otherResult):
        numSamples = self.numSamples
        if self.numSamples > otherResult.numSamples:
            numSamples = otherResult.numSamples
            if self.verbose:
                print("Restricting numSamples to", numSamples)
                sys.stdout.flush()

        sqrtN = numpy.sqrt(numSamples)

        Obs1 = self.results[0:numSamples]
        Obs2 = otherResult.results[0:numSamples]

        Diff = Obs2 - Obs1
        MeanDiff = Diff.mean(dtype = numpy.longdouble)
        StdDiff = numpy.std(Diff, dtype = numpy.longdouble, ddof = 1)
        T = MeanDiff * sqrtN / StdDiff

        if self.verbose:
            print("T value", T)
            sys.stdout.flush()

        if T > 1.833:
            return True
        else:
            return False

Obs1 = [2.3579684928462095711, 2.6966511431047503744, 2.1041541473181432354, 2.7019073416281107448, 2.5738235797042262097, 2.7166646640287669874, 3.1570973772296834006, 2.4292673864464046816, 2.1784786924199874335, 2.3068665222256648137]
Obs2 = [2.1887805567330791, 2.1413028119259576, 2.0204889077292636, 2.4130281192595735, 2.442701709764024, 2.5601243464744949, 3.0302388017521551, 2.1677264377561114, 2.0216193302246714, 2.0081955630917054]

npObs1 = ExperimentResult(Obs1, verbose = False)
print("Obs1", npObs1.reportMean())

npObs2 = ExperimentResult(Obs2, verbose = False)
print("Obs2", npObs2.reportMean())

if npObs1.testDifference(npObs2):
    print("Obs1 > Obs2")
elif npObs2.testDifference(npObs1):
    print("Obs2 > Obs1")
else:
    print("Obs1 == Obs2")
