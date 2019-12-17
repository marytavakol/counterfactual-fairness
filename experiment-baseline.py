import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'fair_classification/')  # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf  # loss funcs that can be optimized subject to various constraints


def test_adult_data(name):
    """ Load the adult data """
    all_data = np.loadtxt(name)
    all_data = ut.add_intercept(all_data)

    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None

    loss_function = lf._logistic_loss
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    mode = 3

    def train_test_classifier():
        w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs_to_cov_thresh, gamma)
        result = ut.test(w, x_test, y_test, x_control_test)
        return result

    AUC = []
    Fair = []
    n_runs = 10

    if mode == 1:

        """ Classify the data while optimizing for accuracy """
        print("== Unconstrained (original) classifier ==")
        # all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
        apply_fairness_constraints = 0
        apply_accuracy_constraint = 0
        sep_constraint = 0

        for run in range(n_runs):
            results = train_test_classifier()
            AUC.append(results[0])
            Fair.append(results[1])

        print("Average AUC: ", np.mean(AUC))
        print("STD-Error AUC: ", np.std(AUC) / np.sqrt(n_runs))

        print("Average P-rule: ", np.mean(Fair))
        print("STD-Error P-rule: ", np.std(Fair) / np.sqrt(n_runs))

    elif mode == 2:

        apply_fairness_constraints = 1  # set this flag to one since we want to optimize accuracy subject to fairness constraints
        apply_accuracy_constraint = 0
        sep_constraint = 0
        print("== Classifier with fairness constraint ==")

        for run in range(n_runs):

            train_data, test_data = train_test_split(all_data, shuffle=True, test_size=0.3)
            x_train = train_data[:, :-2]
            x_control_train = train_data[:, -2]
            y_train = train_data[:, -1]
            x_test = test_data[:, :-2]
            x_control_test = test_data[:, -2]
            y_test = test_data[:, -1]

            results = train_test_classifier()
            AUC.append(results[0])
            Fair.append(results[1])

    elif mode == 3:

        apply_fairness_constraints = 0  # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
        apply_accuracy_constraint = 1  # now, we want to optimize fairness subject to accuracy constraints
        sep_constraint = 0

        for gamma in np.arange(0, 1.1, 0.1):
            print("== Classifier with accuracy constraint == gamma", gamma)

            AUC = []
            Fair = []
            for run in range(n_runs):

                train_data, test_data = train_test_split(all_data, shuffle=True, test_size=0.3)
                x_train = train_data[:, :-2]
                x_control_train = train_data[:, -2]
                y_train = train_data[:, -1]
                x_test = test_data[:, :-2]
                x_control_test = test_data[:, -2]
                y_test = test_data[:, -1]

                results = train_test_classifier()
                AUC.append(results[0])
                Fair.append(results[1])

            print("Average AUC: ", np.mean(AUC))
            print("STD-Error AUC: ", np.std(AUC) / np.sqrt(n_runs))

            print("Average P-rule: ", np.mean(Fair))
            print("STD-Error P-rule: ", np.std(Fair) / np.sqrt(n_runs))

    # """
    # Classify such that we optimize for fairness subject to a certain loss in accuracy
    # In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.
    # """
    # apply_fairness_constraints = 0  # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
    # apply_accuracy_constraint = 1  # now, we want to optimize accuracy subject to fairness constraints
    # sep_constraint = 1  # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
    # gamma = 1000.0
    # print("== Classifier with accuracy constraint (no +ve misclassification) ==")
    # results = train_test_classifier()
    # print(results)

    return




if __name__ == '__main__':

    name = "data/adult-cleaned.dat"
    if len(sys.argv) > 1:
        name = str(sys.argv[1])

    test_adult_data(name)
