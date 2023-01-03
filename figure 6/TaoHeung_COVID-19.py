#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import math
from scipy import stats
from random import random
import numpy as np
import os
import csv


outbreakData = pd.read_csv('data_COVID_TaoHeung.csv')

# Parameters to be estimated
# Additional number of cases per day
clusterSize = [50,50,50]
meanIncubation = [7,7,7]
stdIncubation = [2,2,2]
meanRepoDelay = [2,3,4]
stdRepoDelay = [1,1,1]


# Prior
priorMeanIncu = 7
priorStdIncu = 2
priorRepoDelay = 2
priorStdDelay = 1


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
        
        
# Constant
numCluster = max(outbreakData.hosp_cluster_idx)
clusterExposure = [0,10,20]
numMeanIncubation = 1;
numMeanDelay = 1;
distrIncubation = 'Lognormal';
distrDelay = 'Lognormal';
mainControl = createConstants(numCluster,numMeanIncubation,numMeanDelay,distrIncubation,distrDelay,clusterExposure)
mainControl.priorIncu.describe()


x0 = clusterSize[0:mainControl.numCluster] + meanIncubation[0:mainControl.numMeanIncubation] + stdIncubation[0:mainControl.numMeanIncubation] + meanRepoDelay[0:mainControl.numMeanDelay] + stdRepoDelay[0:mainControl.numMeanDelay]
x0LowerBound = [0] * mainControl.numCluster + [0.01] * mainControl.numMeanIncubation + [0.01] * mainControl.numMeanIncubation + [0.01] * mainControl.numMeanDelay + [0.01] * mainControl.numMeanDelay
x0UpperBound = [200] * mainControl.numCluster + [50] * mainControl.numMeanIncubation + [50] * mainControl.numMeanIncubation + [50] * mainControl.numMeanDelay + [50] * mainControl.numMeanDelay


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
    #         exposure->(incubation)onset->(delay)confirmation
        sData[iiCluster][mainControl.WITHONSET]=[tempData["onset_avg_date"][tempData.onset_avg_date>-999] - tempData["exposure_avg_date"][tempData.onset_avg_date>-999], tempData["confirmation_avg_date"][tempData.onset_avg_date>-999] - tempData["onset_avg_date"][tempData.onset_avg_date>-999]]
        sData[iiCluster][mainControl.WITHOUTONSET]=[tempData["confirmation_avg_date"][tempData.onset_avg_date==-999] - tempData["exposure_avg_date"][tempData.onset_avg_date==-999]]
    return [maxRepoDate, sData]


def lognpara(mean, std):
    sigma = math.sqrt(math.log((std**2)/(mean**2)+1))
    mu = math.log(mean)-sigma**2/2
    return [mu, sigma]


def logPriorIncu(mainControl,meanPar,stdPar):
    logPrior = stats.norm.logpdf(meanPar,5.2,1)+stats.norm.logpdf(stdPar,3.9,1)
    return logPrior


def logPriorDelay(mainControl,meanPar,stdPar):
    logPrior = stats.norm.logpdf(meanPar,5.3,1)+stats.norm.logpdf(stdPar,4.0,1)
    return logPrior


def loglikelihood(parms,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime):
#     parameters
    lengthObsSize = [sum(x) for x in zip(lengthIncu, lengthComb)]
    clusterSize = parms[0:len([i for i in lengthObsSize if i>0])]
    meanIncubation = parms[mainControl.numCluster+mainControl.numMeanIncubation-1]
    stdIncubation = parms[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1]
    meanRepoDelay = parms[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1]
    stdRepoDelay = parms[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1]
#     get distribution parameters
#     incubation
    if mainControl.distrIncubation == "Lognormal":
        [incuParaA,incuParaB] = lognpara(meanIncubation,stdIncubation)
    else:
        print ("Incubation period distribution misspecified.")
#     reporting delay
    if mainControl.distrDelay == "Lognormal":
        [delayParaA,delayParaB] = lognpara(meanRepoDelay,stdRepoDelay);
    else:
        print ("Incubation period distribution misspecified.")
#     max reporting date
    logL = [0]*len(clusterSize)
#     joint pdf by maxRepoDate
    dt = 0.2
    vecRepo = [x/10 for x in range(0, mainControl.maxPossibleReportDate*10+2, 2 )]
    pJoint = []
    for i in range(len(vecRepo)):
        pJoint.append([0]*len(vecRepo))
    for iiReport in range(len(vecRepo)): 
        currRepo = vecRepo[iiReport]
        arrIncu = vecRepo[0:iiReport+1]
        cpIncu = stats.lognorm.cdf([i+dt for i in arrIncu], incuParaB, scale = math.exp(incuParaA)) - stats.lognorm.cdf(arrIncu, incuParaB, scale = math.exp(incuParaA))
        cpRepo = stats.lognorm.cdf([dt+currRepo-i for i in arrIncu], delayParaB, scale = math.exp(delayParaA)) - stats.lognorm.cdf([currRepo-i for i in arrIncu], delayParaB, scale = math.exp(delayParaA))
        pJoint[iiReport][0:iiReport+1] = cpIncu*cpRepo
#     return pJoint
    for iiCluster in range(len(clusterSize)):
        pObs = []
        for column in range(len(pJoint[0:int((untilTime-mainControl.clusterExposure[iiCluster])/dt)])):
            pObs.append(sum(pJoint[0:int((untilTime-mainControl.clusterExposure[iiCluster])/dt)][column]))
        try:
            logLobs = math.log(max(stats.binom.pmf(math.ceil(clusterSize[iiCluster]), math.ceil(clusterSize[iiCluster]+lengthObsSize[iiCluster]), 1-sum(pObs)), 1e-20))-lengthObsSize[iiCluster]*math.log(max(sum(pObs),1e-20))
        except OverflowError:
            logLobs = math.log(max(float('0'), 1e-20))-lengthObsSize[iiCluster]*math.log(max(sum(pObs),1e-20))
        except ValueError: 
            logLobs = math.log(max(float('nan'), 1e-20))-lengthObsSize[iiCluster]*math.log(max(sum(pObs),1e-20))
        logLobsIncu = []
        list = stats.lognorm.pdf(obsIncubation[iiCluster][obsIncubation[iiCluster]>0].values.tolist(), incuParaB, scale = math.exp(incuParaA))
        for x in list:
            if x > 0:
                logLobsIncu.append(math.log(x))
            else:
                logLobsIncu.append(math.log(1e-20))
        logLobsDelay = []
        list = stats.lognorm.pdf(obsDelay[iiCluster][obsDelay[iiCluster]>0].values.tolist(), delayParaB, scale = math.exp(delayParaA))
        for x in list:
            if x > 0:
                logLobsDelay.append(math.log(x))
            else:
                logLobsDelay.append(math.log(1e-20))
        if not obsCombined[iiCluster].empty:
            logLobsCombined = [0]*len(obsCombined[iiCluster])
            for ii in range(len(obsCombined[iiCluster])):
                tempObsComb = obsCombined[iiCluster].to_numpy()[ii]
                logLobsCombined[ii] = math.log(pObs[min(math.ceil(tempObsComb/dt),len(pObs))-1]/dt) if pObs[min(math.ceil(tempObsComb/dt),len(pObs))-1]/dt > 0 else -math.inf
            logL[iiCluster] = logLobs+sum(np.array(logLobsIncu)[np.array(logLobsIncu)>-math.inf])+sum(np.array(logLobsDelay)[np.array(logLobsDelay)>-math.inf])+sum(np.array(logLobsCombined)[np.array(logLobsCombined)>-math.inf])
        else:
            logL[iiCluster] = logLobs+sum(np.array(logLobsIncu)[np.array(logLobsIncu)>-math.inf])+sum(np.array(logLobsDelay)[np.array(logLobsDelay)>-math.inf])    
    return sum(logL)


def mcmcProposal(parameters_c,parameterSteps,lowerRange,upperRange):
    parameters_new = [-999]*len(parameters_c)
    numWhileLoop = 0
    rand = []
    for i in range(len(parameters_c)):
        rand.append((random()-0.5)*2)
    while (pd.Series(parameters_new) < lowerRange).any() or (pd.Series(parameters_new) > upperRange).any():
        numWhileLoop +=1
        parameters_new = parameters_c + np.array(parameters_new)*np.array(rand)
        parameters_new = np.where(np.array(parameters_new)<lowerRange, np.array(lowerRange)+(np.array(lowerRange)-np.array(parameters_new)), np.array(parameters_new))
        parameters_new = np.where(np.array(parameters_new)>upperRange, np.array(upperRange)-(np.array(parameters_new)-np.array(upperRange)), np.array(parameters_new))
        if numWhileLoop >1000:
            break
    return parameters_new


def mcmcParallel(mcdir,mcSteps,parameters,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime,parameterSteps,lowerLimit,upperLimit):
    goodMC = False
    pAccept = [0.5]*len(parameters)
    numWhileLoop = 0
    numPars = len(parameters)
    
    while goodMC == False:
        numWhileLoop += 1
        iPar = []
        for i in range(mcSteps):
            iPar.append([0]*numPars)
        iiLogL = [0]*mcSteps
        nAccept = [0]*numPars
        parameterSteps[np.array(pAccept)>0.7] = parameterSteps[np.array(pAccept)>0.7]*1.2
        parameterSteps[np.array(pAccept)<0.3] = parameterSteps[np.array(pAccept)<0.3]/1.25
        parameterSteps = [parameterSteps,np.array(upperLimit)-np.array(lowerLimit)]
        parameterSteps = np.array(parameterSteps).min(axis=0)
    #     current loglikelihood
        parameters_c = parameters
        logLikelihood = loglikelihood(parameters_c,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime)
        logIncuPrior = logPriorIncu(mainControl,parameters_c[mainControl.numCluster+mainControl.numMeanIncubation-1],parameters_c[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1])
        logDelayPrior = logPriorDelay(mainControl,parameters_c[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1],parameters_c[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1])
        for tt in range(mcSteps):
            tempNew = mcmcProposal(parameters_c,parameterSteps,lowerLimit,upperLimit)
            for ii in range(numPars):
                parameters_mc = parameters_c.copy()
                parameters_mc[ii] = tempNew[ii]
                logLikelihood_new = loglikelihood(parameters_mc,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime)
                logIncuPrior_new = logPriorIncu(mainControl,parameters_mc[mainControl.numCluster+mainControl.numMeanIncubation-1],parameters_mc[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1])
                logDelayPrior_new = logPriorDelay(mainControl,parameters_mc[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1],parameters_mc[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1])
        #         alpha
                try:
                    ans = math.exp(logLikelihood_new+logIncuPrior_new+logDelayPrior_new-logLikelihood-logIncuPrior-logDelayPrior)
                except OverflowError:
                    ans = float('inf')
                alpha = min(1,ans)
                uu = random()
                if uu <= alpha:
                    parameters_c = parameters_mc.copy()
                    logLikelihood = logLikelihood_new
                    logIncuPrior = logIncuPrior_new
                    logDelayPrior = logDelayPrior_new
                    nAccept[ii] += 1
            iiLogL[tt] = logLikelihood
            iPar[tt] = parameters_c.copy()
        #     check MCMC
            if (tt+1) % (mcSteps/100) == 0:
                pAccept = np.array(nAccept)/(tt+1)
                with open(os.path.join(path,"parameter_step.csv"),"w") as f:
                    writer = csv.writer(f)
                    writer.writerow(parameterSteps)
                with open(os.path.join(path,"mcmc_res.csv"),"w",newline='') as f:
                    writer = csv.writer(f)
                    for i in range(mcSteps):
                        writer.writerow(iPar[i])
                with open(os.path.join(path,"log_likelihood.csv"),"w") as f:
                    writer = csv.writer(f)
                    writer.writerow(iiLogL)
                print("MCMC")
                print((tt+1)/mcSteps)
                print("Acceptance probability")
                print(pAccept)
                if (pd.Series(pAccept) > 0.7).any() or (pd.Series(pAccept) < 0.3).any():
                    goodMC = False
                    if numWhileLoop < 20:
                        if (tt+1)/mcSteps >= 0.1:
                            break
                    else:
                        goodMC = True
                else:
                    goodMC = True

    return [iPar, iiLogL]

# Until time
timeArr = list(range(int(max(outbreakData.confirmation_avg_date)),12,-1))
for iiTimeArr in range(len(timeArr)):
    untilTime = timeArr[iiTimeArr]
    [maxRepoDate,sData] = sortData(outbreakData[outbreakData.confirmation_avg_date<=untilTime], mainControl)
#     create directory
    mcdir = "{}_{}_{}_{}_until_{}".format(distrIncubation, numMeanIncubation, distrDelay, numMeanDelay, untilTime)
#     reformat data
    obsIncubation = []
    obsDelay = []
    obsCombined = []
    lengthIncu = []
    lengthComb = []
    for iiCluster in range(mainControl.numCluster):
        obsIncubation.append([])
        obsDelay.append([])
        obsCombined.append([])
        lengthIncu.append(0)
        lengthComb.append(0)
        clusterData = sData[iiCluster]
        iiClusterSize = clusterSize[iiCluster]
        # With onset
        clusterWithOnset = clusterData[mainControl.WITHONSET]
        obsIncubation[iiCluster] = clusterWithOnset[0]
        obsDelay[iiCluster] = clusterWithOnset[1]
        lengthIncu[iiCluster] = len(clusterWithOnset[0])
        # Without onset
        clusterWithoutOnset = clusterData[mainControl.WITHOUTONSET]
        obsCombined[iiCluster] = clusterWithoutOnset[0]
        lengthComb[iiCluster] = len(clusterWithoutOnset[0])
    # test
    print(-loglikelihood(x0,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime))
    # inference
    mcdir = "mcmc_result_TaoHeung/{}_{}_{}_{}_until_{}".format(distrIncubation, numMeanIncubation, distrDelay, numMeanDelay, untilTime)
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, mcdir)
    if not os.path.isdir(mcdir):
        os.mkdir(path)
    if os.path.isfile(os.path.join(path, "mcmc_res.csv")):
        lastRec = pd.read_csv(os.path.join(path, "mcmc_res.csv"),header=None)
        lastRec = lastRec[lastRec.iloc[:,0]>0]
        xfmin = lastRec.iloc[-1,:]
    else:
        xfmin = x0
#     MCMC
    mcSteps = 10000
    if os.path.isfile(os.path.join(path, "parameter_step.csv")): 
        stepSize = pd.read_csv(os.path.join(path, "parameter_step.csv"),header=None)
        stepSize = stepSize.iloc[0,:]
    else: 
        stepSize = 0.2*np.array(xfmin)
        stepSize[stepSize<0.0001] = 0.0001
    out = mcmcParallel(mcdir,mcSteps,xfmin,obsIncubation,obsDelay,obsCombined,lengthIncu,lengthComb,mainControl,untilTime,stepSize,x0LowerBound,x0UpperBound)
