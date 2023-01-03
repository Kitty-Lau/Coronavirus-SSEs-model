#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


outbreakData = pd.read_csv('data_SouthKorea_MERS.csv')


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
        
        self.priorIncu = pd.read_csv('incub_prior_MERS.csv')
        self.priorDelay = pd.read_csv('delay_prior_MERS.csv')


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

# plot
timeArr = list(range(11, int(max(outbreakData.confirmation_avg_date))+1,1))
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
    mcdir = "mcmc_result_SouthKorea/{}_{}_{}_{}_until_{}".format(distrIncubation, numMeanIncubation, distrDelay, numMeanDelay, untilTime)
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


fig, ax = plt.subplots(4, 1,  figsize=(11,23),sharex=True)
fig.suptitle("South Korea MERS Clusters", fontsize=21, y=0.91)
fig.supxlabel("Day since the exposure of Cluster 1", fontsize=21, y=0.1)
fig.supylabel("Case count", x=0.01, fontsize=21)
colors = ["tab:blue", "tab:orange", "tab:green"]
labels = ["Cluster 1", "Cluster 2", "Cluster 3"]
lab = ["Time of Confirmation for Cluster 1", "Time of Confirmation for Cluster 2", "Time of Confirmation for Cluster 3"]
ylims = [[0,50], [0,250], [0,80]]
binBoundaries = np.linspace(0,45,45)
ax[0].hist(histConf, bins=binBoundaries, ec="k", stacked=True, color = colors)
ax[0].legend(lab, loc="upper right", frameon=False, fontsize = 12)
for iiCluster in range(numCluster):
    ax[iiCluster+1].hist(histConf[iiCluster],bins=binBoundaries, ec="k", color=colors[iiCluster], cumulative=True, label='Cumulative confirmed cases')
    ax[iiCluster+1].hlines(y=lengthAll[iiCluster], xmin=0, xmax=len(binBoundaries), color=colors[iiCluster], linestyles='--', label='Final cluster size')
    ax[iiCluster+1].set_ylim(ylims[iiCluster])
    ax[iiCluster+1].set_title(labels[iiCluster])
    rec = [x for x in recPatch[iiCluster] if x != []]
    rec = np.array(rec).T
    ax[iiCluster+1].plot(rec[0],rec[1], color=colors[iiCluster], label="Real-time estimation")
    ax[iiCluster+1].fill_between(rec[0], rec[2], rec[3], color=colors[iiCluster], alpha=0.2, label="95% credible interval")
    ax[iiCluster+1].legend(loc="upper right", frameon=False, fontsize = 13)
plt.savefig('South Korea MERS Clusters.png', dpi=500)
