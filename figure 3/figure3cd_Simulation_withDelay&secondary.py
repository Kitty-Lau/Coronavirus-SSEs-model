#!/usr/bin/env python
# coding: utf-8


from scipy import stats
from numpy import random
from random import random as rd
import os
import pandas as pd
import numpy as np
import math
import statistics


# Constant
numPriorVec = [50,100]
numPointSourceVec = [30,50,100,200]
numScn = 1000
numParms = 7
pre = 0.6
# Latin hypercube3 sampling
sampler = stats.qmc.LatinHypercube(d=5)
lhSample = sampler.random(n=1000)
meanIncuRange = [5,8]
stdIncuRange = [2,5]
meanDelayRange = [3,7]
stdDelayRange = [2,5]
percInfRange = [0.05,0.95]


if not os.path.isfile("figure3cd_rec_latin_hypercube_samples.csv"):
    rec = [meanIncuRange[0]+(meanIncuRange[1]-meanIncuRange[0])*lhSample[:,0], stdIncuRange[0]+(stdIncuRange[1]-stdIncuRange[0])*lhSample[:,1], meanDelayRange[0]+(meanDelayRange[1]-meanDelayRange[0])*lhSample[:,2], stdDelayRange[0]+(stdDelayRange[1]-stdDelayRange[0])*lhSample[:,3], percInfRange[0]+(percInfRange[1]-percInfRange[0])*lhSample[:,4]]
    rec = pd.DataFrame(rec).T
    rec.to_csv("figure3cd_rec_latin_hypercube_samples.csv", header=None, index=False)
else:
    rec = pd.read_csv("figure3cd_rec_latin_hypercube_samples.csv",header=None)


# Generation time
meanGen = [5,7]
stdGen = [5,7]
if not os.path.isfile("figure3cd_gen_latin_hypercube_samples.csv"):
    gen = [meanGen[0]+(meanGen[1]-meanGen[0])*lhSample[:,5], stdGen[0]+(stdGen[1]-stdGen[0])*lhSample[:,6]]
    gen = pd.DataFrame(gen).T
    gen.to_csv("figure3cd_gen_latin_hypercube_samples.csv", header=None, index=False)
else:
    gen = pd.read_csv("figure3cd_gen_latin_hypercube_samples.csv",header=None)
    

kkText = 0
recComplete = pd.DataFrame()
genComplete = pd.DataFrame()
for iiPrior in range(len(numPriorVec)):
    numPrior = numPriorVec[iiPrior]
    for iiSource in range(len(numPointSourceVec)):
        numPointSource = numPointSourceVec[iiSource]
        for iiScn in range(numScn):
            scnIdx = iiScn
            recComplete = recComplete.append(rec.loc[iiScn],ignore_index=True)
            recComplete.loc[kkText,5] = numPrior
            recComplete.loc[kkText,6] = numPointSource
            genComplete = genComplete.append(gen.loc[iiScn],ignore_index=True)
            kkText += 1


# constant parameters
numCluster = 1
clusterExposure = 0
numMeanIncubation = 1
numMeanDelay = 1
distrIncubation = 'Lognormal'
distrDelay = 'Lognormal'
numBoot = 1000
numMeanGen = 1
distrGen = 'Lognormal'


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
        
        self.priorIncu = []
        self.priorDelay = []
        self.dateUntil = int


def lognpara(mean, std):
    sigma = math.sqrt(math.log((std**2)/(mean**2)+1))
    mu = math.log(mean)-sigma**2/2
    return [mu, sigma]
    

def logPriorIncu(mainControl,meanPar,stdPar):
    logPrior = stats.norm.logpdf(meanPar,mainControl.priorIncu["mean"].mean(),scale=mainControl.priorIncu["mean"].std())+stats.norm.logpdf(stdPar,mainControl.priorIncu["std"].mean(),scale=mainControl.priorIncu["std"].std())
    return logPrior


def logPriorDelay(mainControl,meanPar,stdPar):
    logPrior = stats.norm.logpdf(meanPar,mainControl.priorDelay["mean"].mean(),scale=mainControl.priorDelay["mean"].std())+stats.norm.logpdf(stdPar,mainControl.priorDelay["std"].mean(),scale=mainControl.priorDelay["std"].std())
    return logPrior


def mcmcProposal(parameters_c,parameterSteps,lowerRange,upperRange):
    parameters_new = [-999]*len(parameters_c)
    numWhileLoop = 0
    rand = []
    for i in range(len(parameters_c)):
        rand.append((rd()-0.5)*2)
    while (pd.Series(parameters_new) < lowerRange).any() or (pd.Series(parameters_new) > upperRange).any():
        numWhileLoop +=1
        parameters_new = parameters_c + np.array(parameters_new)*np.array(rand)
        parameters_new = np.where(np.array(parameters_new)<lowerRange, np.array(lowerRange)+(np.array(lowerRange)-np.array(parameters_new)), np.array(parameters_new))
        parameters_new = np.where(np.array(parameters_new)>upperRange, np.array(upperRange)-(np.array(parameters_new)-np.array(upperRange)), np.array(parameters_new))
        if numWhileLoop >1000:
            break
    return parameters_new


def loglikelihood(parms,incuData,delayData,mainControl):
    clusterSize = parms[0:mainControl.numCluster]
    meanIncubation = parms[mainControl.numCluster+mainControl.numMeanIncubation-1]
    stdIncubation = parms[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1]
    meanRepoDelay = parms[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1]
    stdRepoDelay = parms[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1]
    if mainControl.distrIncubation == "Lognormal":
        [incuParaA,incuParaB] = lognpara(meanIncubation,stdIncubation)
    else:
        print ("Incubation period distribution misspecified.")
    #     reporting delay
    if mainControl.distrDelay == "Lognormal":
        [delayParaA,delayParaB] = lognpara(meanRepoDelay,stdRepoDelay);
    else:
        print ("Delay period distribution misspecified.")
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
        for column in range(len(pJoint[0:int(mainControl.dateUntil/dt)])):
            pObs.append(sum(pJoint[0:int(mainControl.dateUntil/dt)][column]))
        if not pObs:
            pObs.append(1e-20)
        try:
            logLobs = math.log(max(stats.binom.pmf(math.ceil(clusterSize[iiCluster]), math.ceil(clusterSize[iiCluster]+len(incuData)), 1-sum(pObs)), 1e-20))-len(incuData)*math.log(max(sum(pObs),1e-20))
        except OverflowError:
            logLobs = math.log(max(float('0'), 1e-20))-len(incuData)*math.log(max(sum(pObs),1e-20))
        except ValueError: 
            logLobs = math.log(max(float('nan'), 1e-20))-len(incuData)*math.log(max(sum(pObs),1e-20))
        logLobsIncu = []
        list = stats.lognorm.pdf(incuData, incuParaB, scale = math.exp(incuParaA))
        for x in list:
            if x > 0:
                logLobsIncu.append(math.log(x))
            else:
                logLobsIncu.append(math.log(1e-20))
        logLobsDelay = []
        list = stats.lognorm.pdf(delayData, delayParaB, scale = math.exp(delayParaA))
        for x in list:
            if x > 0:
                logLobsDelay.append(math.log(x))
            else:
                logLobsDelay.append(math.log(1e-20))
        logL[iiCluster] = logLobs+sum(np.array(logLobsIncu))+sum(np.array(logLobsDelay))    
    return sum(logL)


def mcmcParallel(mcSteps,parameters,incuData,delayData,mainControl,parameterSteps,lowerLimit,upperLimit):
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
        logLikelihood = loglikelihood(parameters_c,incuData,delayData,mainControl)
        logIncuPrior = logPriorIncu(mainControl,parameters_c[mainControl.numCluster+mainControl.numMeanIncubation-1],parameters_c[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1])
        logDelayPrior = logPriorDelay(mainControl,parameters_c[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1],parameters_c[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1])
        for tt in range(mcSteps):
            tempNew = mcmcProposal(parameters_c,parameterSteps,lowerLimit,upperLimit)
            for ii in range(numPars):
                parameters_mc = parameters_c.copy()
                parameters_mc[ii] = tempNew[ii]
                logLikelihood_new = loglikelihood(parameters_mc,incuData,delayData,mainControl)
                logIncuPrior_new = logPriorIncu(mainControl,parameters_mc[mainControl.numCluster+mainControl.numMeanIncubation-1],parameters_mc[mainControl.numCluster+mainControl.numMeanIncubation+mainControl.numMeanIncubation-1])
                logDelayPrior_new = logPriorDelay(mainControl,parameters_mc[mainControl.numCluster+2*mainControl.numMeanIncubation+mainControl.numMeanDelay-1],parameters_mc[mainControl.numCluster+2*mainControl.numMeanIncubation+2*mainControl.numMeanDelay-1])
        #         alpha
                try:
                    ans = math.exp(logLikelihood_new+logIncuPrior_new+logDelayPrior_new-logLikelihood-logIncuPrior-logDelayPrior)
                except OverflowError:
                    ans = float('inf')
                alpha = min(1,ans)
                uu = rd()
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


for iiScn in range(len(recComplete)):
    if not os.path.isfile("figure3cd_mcmc_res/mcmc_res_all_rec_{}.csv".format(iiScn+1)):
#         create constant parameters
        mainControl = createConstants(numCluster,numMeanIncubation,numMeanDelay,distrIncubation,distrDelay,clusterExposure)
        simParms = recComplete.loc[iiScn]
        genParms = genComplete.loc[iiScn]
#     get distribution parameters
#     incubation
        if mainControl.distrIncubation == "Lognormal":
            [incuParaA,incuParaB] = lognpara(simParms[0],simParms[1])
        else:
            print ("Incubation period distribution misspecified.")
#     reporting delay
        if mainControl.distrDelay == "Lognormal":
            [delayParaA,delayParaB] = lognpara(simParms[2],simParms[3])
        else:
            print ("Delay period distribution misspecified.")
#     generation time
        if distrGen == "Lognormal":
            [genParaA,genParaB] = lognpara(genParms[0],genParms[1])
        else:
            print ("Generation time distribution misspecified.")
        rawPriorIncuData = random.lognormal(incuParaA,incuParaB,int(simParms[5]))
        rawIncu = random.lognormal(incuParaA,incuParaB,int(simParms[6]))
        rawPriorDelayData = random.lognormal(delayParaA,delayParaB,int(simParms[5]))
        rawDelayData = random.lognormal(delayParaA,delayParaB,int(simParms[6]))
        rawGen = random.lognormal(genParaA,genParaB,int(simParms[6]))
        rawIncuData = rawIncu.copy()
        for i in range (math.ceil(simParms[6]*pre)):
            rawIncuData[i]=rawIncu[i]+rawGen[i]
        # percentage of case
        caseUntil = math.ceil(len(rawIncuData)*simParms[4])
        orderIncuDelay = rawIncuData+rawDelayData
        orderIncuDelay.sort()
        maxTime = max(orderIncuDelay)
        dateUntil = orderIncuDelay[caseUntil-1]
        mainControl.dateUntil = dateUntil
        # bootstrapping
        # incubation data
        # observed til the percentage
        incuData = rawIncuData[rawIncuData+rawDelayData<=dateUntil]
        if mainControl.distrIncubation == "Lognormal":
            priorIncuData = pd.DataFrame()
            for _ in range(numBoot):
                x = random.choice(rawPriorIncuData, size=len(rawPriorIncuData), replace=True)
                shape, loc, scale = stats.lognorm.fit(x,floc=0)
                priorIncuData = priorIncuData.append({"mean":stats.lognorm.mean(shape,scale=scale),"std":stats.lognorm.std(shape,scale=scale)},ignore_index=True)
        else:
            print ("Incubation period distribution misspecified.")
        mainControl.priorIncu = priorIncuData.copy()
        # delay data
        delayData = rawDelayData[rawIncuData+rawDelayData<=dateUntil]
        if mainControl.distrDelay == "Lognormal":
            priorDelayData = pd.DataFrame()
            for _ in range(numBoot):
                x = random.choice(rawPriorDelayData, size=len(rawPriorDelayData), replace=True)
                shape, loc, scale = stats.lognorm.fit(x,floc=0)
                priorDelayData = priorDelayData.append({"mean":stats.lognorm.mean(shape,scale=scale),"std":stats.lognorm.std(shape,scale=scale)},ignore_index=True)
        else:
            print ("Delay period distribution misspecified.")                    
        mainControl.priorDelay = priorDelayData.copy()
        # inference constant
        addSize = 30
        meanIncu = 5
        stdIncu = 6
        meanDelay = 2
        stdDelay = 1
        x0 = [addSize,meanIncu,stdIncu,meanDelay,stdDelay]
        x0LowerBound = [0.01,0.01,0.01,0.01,0.01]
        x0UpperBound = [200,50,50,50,50]
        # test
        print(-loglikelihood(x0,incuData,delayData,mainControl))
        xfmin = x0
        mcSteps = 2000
        stepSize = 0.2*np.array(xfmin)
        stepSize[stepSize<0.2] = 0.2
        out = mcmcParallel(mcSteps,xfmin,incuData,delayData,mainControl,stepSize,x0LowerBound,x0UpperBound)
        # record
        burnIn = 0.5
        mcRes = [column[int(len(out[0])*burnIn):] for column in out][0]
        allRecTable = pd.DataFrame()
        # scnRec
        allRecTable[['mean_incubation','std_incubation','mean_delay','std_delay','percent','prior_size','true_size']] = [simParms]
        # iiRec
        allRecTable[['date_until','max_time_span','num_infer_incubation','num_infer_delay']] = [dateUntil,maxTime,len(incuData),len(delayData)]
        allRecTable[['mode_add_size']] = statistics.mode([round(column[0]) for column in out[0]])
        # mcRes
        allRecTable[['add_size_50','add_size_2_5','add_size_97_5','mean_incubation_50','mean_incubation_2_5','mean_incubation_97_5','std_incubation_50','std_incubation_2_5','std_incubation_97_5','mean_delay_50','mean_delay_2_5','mean_delay_97_5','std_delay_50','std_delay_2_5','std_delay_97_5']] = np.reshape(np.quantile(mcRes,[0.5, 0.025, 0.975],axis=0),(1,-1),order="F")
        allRecTable.to_csv("figure3cd_mcmc_res/mcmc_res_all_rec_{}.csv".format(iiScn+1), index=False)
