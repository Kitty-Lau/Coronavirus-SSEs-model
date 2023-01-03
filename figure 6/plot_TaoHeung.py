#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


outbreakData = pd.read_csv('data_COVID_TaoHeung.csv')


class createConstants: 
    def __init__(self, numCluster, numMeanIncubation, numMeanDelay, distrIncubation, distrDelay, clusterExposure):
        self.numCluster = numCluster;
        self.numMeanIncubation = numMeanIncubation;
        self.numMeanDelay = numMeanDelay;
        self.distrIncubation = distrIncubation;
        self.distrDelay = distrDelay;
        self.clusterExposure = clusterExposure;
        
        self.maxPossibleReportDate = 90;
        self.WITHONSET = 0;
        self.WITHOUTONSET = 1;


# constant
numCluster = max(outbreakData.hosp_cluster_idx)
lengthAll = [0] * numCluster
histConf = []
histOns = []
for iiCluster in range(numCluster):
    clusterData = outbreakData[outbreakData.hosp_cluster_idx == iiCluster+1]
    lengthAll[iiCluster] = len(clusterData)
    histConf.append(clusterData["confirmation_avg_date"])
    histOns.append(clusterData["onset_avg_date"])
# clusterData
clusterExposure = [0,10,20]
numMeanIncubation = 1
numMeanDelay = 1
distrIncubation = 'Lognormal'
distrDelay = 'Lognormal'
mainControl = createConstants(numCluster,numMeanIncubation,numMeanDelay,distrIncubation,distrDelay,clusterExposure);


def sortData(data, mainControl):
    sData = []
    maxRepoDate = []
    for iiCluster in range(mainControl.numCluster):
        tempData = data[data.hosp_cluster_idx == iiCluster+1]
        sData.append([[[],[]],[]])
        maxRepoDate.append(0)
        if not tempData.confirmation_avg_date.empty:
            if max(tempData.confirmation_avg_date):
                maxRepoDate[iiCluster] = max(tempData.confirmation_avg_date)
        sData[iiCluster][mainControl.WITHONSET]=[tempData["onset_avg_date"][tempData.onset_avg_date>-999] - tempData["exposure_avg_date"][tempData.onset_avg_date>-999], tempData["confirmation_avg_date"][tempData.onset_avg_date>-999] - tempData["onset_avg_date"][tempData.onset_avg_date>-999]]
        sData[iiCluster][mainControl.WITHOUTONSET]=[tempData["confirmation_avg_date"][tempData.onset_avg_date==-999] - tempData["exposure_avg_date"][tempData.onset_avg_date==-999]]
    return [maxRepoDate, sData]


timeArr = list(range(12, int(max(outbreakData.confirmation_avg_date))+1,1))
burnIn = 0.3
recPatch = []
for iiCluster in range(numCluster):
    recPatch.append([])
    for iiTimeArr in range(len(timeArr)):
        recPatch[iiCluster].append([])
for iiTimeArr in range(len(timeArr)):
    untilTime = timeArr[iiTimeArr]
    [maxRepoDate,sData] = sortData(outbreakData[outbreakData.confirmation_avg_date<=untilTime], mainControl)
    lengthObs=[]
    for iiCluster in range(len(sData)):
        clusterData = sData[iiCluster]
        lengthObs.append(len(clusterData[mainControl.WITHONSET][0])+ len(clusterData[mainControl.WITHOUTONSET][0]))
    mcdir = "mcmc_result_TaoHeung/{}_{}_{}_{}_until_{}".format(distrIncubation, numMeanIncubation, distrDelay, numMeanDelay, untilTime)
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, mcdir)
    if os.path.isfile(os.path.join(path, "mcmc_res.csv")):
        mcmcRes = pd.read_csv(os.path.join(path, "mcmc_res.csv"),header=None)
        mcmcRes = mcmcRes[mcmcRes.iloc[:,0]>0]
        mcmcRes = mcmcRes.iloc[int(len(mcmcRes.iloc[:,0])*burnIn):,]
        for iiCluster in range(numCluster):
            if len(histConf[iiCluster][histConf[iiCluster]<untilTime])>1:
                pSize = mcmcRes.iloc[:,iiCluster].quantile([0.5, 0.025, 0.975])
                plotSize = [pSize.iloc[0], pSize.iloc[0]-pSize.iloc[1], pSize.iloc[2]-pSize.iloc[0]]
                recPatch[iiCluster][iiTimeArr] = [untilTime, lengthObs[iiCluster]+pSize.iloc[0], lengthObs[iiCluster]+pSize.iloc[1], lengthObs[iiCluster]+pSize.iloc[2]]  


fig, ax = plt.subplots(3, 1,  figsize=(11,23),sharex=True)
fig.suptitle("TaoHeung COVID-19 Cluster", fontsize=21, y=0.91)
fig.supxlabel("Day since the exposure", y=0.1,fontsize=21)
fig.supylabel("Case count", x=0.01,fontsize=21)
ylims = [[0,100], [0,250], [0,80]]
binBoundaries = np.linspace(0,40,40)
ax[0].hist(histOns[0], bins=binBoundaries, ec="k", stacked=True, color = "tab:blue", label='Time of Symptom Onset')
ax[0].legend(loc="upper right", frameon=False,fontsize = 16)
ax[1].hist(histConf[0], bins=binBoundaries, ec="k", stacked=True, color = "tab:blue", label='Time of Confirmation')
ax[1].legend(loc="upper right", frameon=False, fontsize = 16)
ax[2].hist(histConf[0],bins=binBoundaries, ec="k", color="tab:blue", cumulative=True, label='Cumulative confirmed cases')
ax[2].hlines(y=lengthAll[0], xmin=0, xmax=len(binBoundaries), color="tab:blue", linestyles='--', label='Final cluster size')
ax[2].set_ylim(ylims[0])
rec = [x for x in recPatch[0] if x != []]
rec = np.array(rec).T
ax[2].plot(rec[0],rec[1], color="tab:blue", label="Real-time estimation")
ax[2].fill_between(rec[0], rec[2], rec[3], color="tab:blue", alpha=0.2, label="95% credible interval")
ax[2].legend(loc="upper right", frameon=False, fontsize = 12)
plt.savefig('TaoHeung COVID-19 Cluster.png', dpi=500)
