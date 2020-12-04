from assign5 import BoostClassifier
import labfuns
from labfuns import DecisionTreeClassifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dset = "vowel"
    # print("Vowel Data set:")
    # print("Without Boosting:")
    # labfuns.testClassifier(DecisionTreeClassifier(), dataset=dset,split=0.7)
    # print("With Boosting:")
    # labfuns.testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset=dset,split=0.7)
    #==================================
    # MAKE PLOTS
    #==================================
    # print("Without Boosting:")
    # labfuns.plotBoundary(DecisionTreeClassifier(), dataset=dset,split=0.7)

    # ax = plt.gca()
    # ax.set_xlim([-4, 4])
    # ax.set_ylim([-1.5, 1.5])
    # plt.title("Without boosting: Dtree on %s" % dset)
    # plt.savefig("dtree_%s_70.png" % dset, bbox_inches='tight')
    # plt.clf()
    # print("With Boosting:")
    labfuns.plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset=dset,split=0.7)
    # # ax = plt.gca()
    # # ax.set_xlim([-4, 4])
    # # ax.set_ylim([-1.5, 1.5])
    # plt.title("With boosting: Dtree on %s" % dset)
    # plt.savefig("dtree_%s_70_wb.png" % dset, bbox_inches='tight')
    # plt.clf()
