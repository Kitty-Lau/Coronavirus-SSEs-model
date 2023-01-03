#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns


# Constant
numPriorVec = [50,100]
numPointSourceVec = [30,50,100,200]
numScn = 1000
numParms = 3
# Latin hypercube3 sampling
rec = pd.read_csv("figure3ab_latin_hypercube_samples.csv",header=None)


kkText = 0
recComplete = pd.DataFrame()
for iiPrior in range(len(numPriorVec)):
    numPrior = numPriorVec[iiPrior]
    for iiSource in range(len(numPointSourceVec)):
        numPointSource = numPointSourceVec[iiSource]
        for iiScn in range(numScn):
            scnIdx = iiScn
            recComplete = recComplete.append(rec.loc[iiScn],ignore_index=True)
            recComplete.loc[kkText,5] = numPrior
            recComplete.loc[kkText,6] = numPointSource
            kkText += 1


# Combine results
resultRec = pd.DataFrame()
for iiScn in range(len(recComplete)):
    if os.path.isfile("figure3ab_rec_summary/mcmc_res_all_rec_{}.csv".format(iiScn+1)):
        tempRec = pd.read_csv("figure3ab_rec_summary/mcmc_res_all_rec_{}.csv".format(iiScn+1))
        resultRec = resultRec.append(tempRec,ignore_index=True)
# resultRec
resultRec.to_csv("figure3ab_rec_summary.csv",index=False)


def summaryStats (resultRec):
#     summary stats
    resultRec["est_size_50"] = resultRec["num_infer_incubation"]+resultRec["add_size_50"]
    resultRec["est_size_2_5"] = resultRec["num_infer_incubation"]+resultRec["add_size_2_5"]
    resultRec["est_size_97_5"] = resultRec["num_infer_incubation"]+resultRec["add_size_97_5"]
    resultRec["percent_time"] = resultRec["date_until"]/resultRec["max_time_span"]
    resultRec["percent_case"] = resultRec["num_infer_incubation"]/resultRec["true_size"]
    resultRec["rel_err_50"] = (resultRec["est_size_50"]-resultRec["true_size"])/resultRec["true_size"]
    resultRec["rel_err_2_5"] = (resultRec["est_size_2_5"]-resultRec["true_size"])/resultRec["true_size"]
    resultRec["rel_err_97_5"] = (resultRec["est_size_97_5"]-resultRec["true_size"])/resultRec["true_size"]
    if 'mean_delay' in resultRec.columns:
        resultRec["fold_true_incu_delay"] = resultRec["date_until"]/(resultRec["mean_incubation"]+resultRec["mean_delay"])
    else: 
        resultRec["fold_true_incu_delay"] = resultRec["date_until"]/(resultRec["mean_incubation"])
    resultRec["rel_err_mode"] = (resultRec["num_infer_incubation"]+resultRec["mode_add_size"]-resultRec["true_size"])/resultRec["true_size"]
    resultRec["true_remain"] = resultRec["true_size"]-resultRec["num_infer_incubation"]
    resultRec["rel_err_remain_50"] = (resultRec["add_size_50"].apply(np.ceil)-resultRec["true_remain"])/resultRec["true_remain"]
    return resultRec
    

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def fig_err_outbreak_time(resultRec):
    boxplotGroup = pd.DataFrame()
    boxplotGroup["Relative error"]=resultRec["rel_err_mode"]
    boxplotGroup["Time as percent of the time span of outbreak"] = pd.DataFrame(np.digitize(resultRec["percent_time"],[0.05,0.2,0.4,0.5,0.6,0.8,0.95]))
    boxplotGroup['Time as percent of the time span of outbreak'] = boxplotGroup['Time as percent of the time span of outbreak'].replace([1,2,3,4,5,6],["5-20%","20-40%","40-50%","50-60%","60-80%","80-95%"])
    boxplotGroup["Simulated cluster size"]=resultRec["true_size"]
    boxplotGroup = boxplotGroup.astype({'Simulated cluster size':'int'})

    fig = plt.figure(figsize=(10,6.5))
    plt.ylim(-1, 2)
    sns.set_theme(style="white")
    ax = sns.boxplot(x="Time as percent of the time span of outbreak", y="Relative error", data=boxplotGroup, hue="Simulated cluster size",linewidth=0.8,width=0.7,order=["5-20%","20-40%","40-50%","50-60%","60-80%","80-95%"],showfliers = False, palette="muted")
    ax.axhline(0,linestyle='dashed')
    ax.legend(title='Simulated cluster size', loc='upper right')
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='22') # for legend title
    ax.set_xlabel(xlabel="Time as percent of the time span of outbreak",fontsize=22)
    ax.set_ylabel(ylabel="Relative error",fontsize=22)
    ax.tick_params(labelsize=22)
    sns.despine()
    adjust_box_widths(fig, 0.8)
    plt.savefig('figure2a', dpi=500)
    
# percentage case
def fig_err_outbreak_case(resultRec):
    boxplotGroup = pd.DataFrame()
    boxplotGroup["Relative error"]=resultRec["rel_err_mode"]
    boxplotGroup["Percent of observed cases"] = pd.DataFrame(np.digitize(resultRec["percent_case"],[0.05,0.2,0.4,0.5,0.6,0.8,0.95]))
    boxplotGroup['Percent of observed cases'] = boxplotGroup['Percent of observed cases'].replace([1,2,3,4,5,6],["5-20%","20-40%","40-50%","50-60%","60-80%","80-95%"])
    boxplotGroup["Simulated cluster size"]=resultRec["true_size"]
    boxplotGroup = boxplotGroup.astype({'Simulated cluster size':'int'})

    fig = plt.figure(figsize=(10,6.5))
    plt.ylim(-1, 3)
    sns.set_theme(style="white")
    ax = sns.boxplot(x="Percent of observed cases", y="Relative error", data=boxplotGroup, hue="Simulated cluster size",linewidth=0.8,width=0.7,order=["5-20%","20-40%","40-50%","50-60%","60-80%","80-95%"],showfliers = False, palette="muted")
    ax.axhline(0,linestyle='dashed')
    ax.legend(title='Simulated cluster size', loc='upper right')
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='22') # for legend title
    ax.set_xlabel(xlabel="Percent of observed cases",fontsize=22)
    ax.set_ylabel(ylabel="Relative error",fontsize=22)
    ax.tick_params(labelsize=22)
    sns.despine()
    adjust_box_widths(fig, 0.8)
    plt.savefig('figure2b.png', dpi=500)
    

# plots
resultRec = pd.read_csv("figure3ab_rec_summary.csv")
resultRec = summaryStats(resultRec.iloc[:,:18])


fig_err_outbreak_time(resultRec.loc[(resultRec["true_remain"]/resultRec["true_size"]>=0) & (resultRec["true_remain"]/resultRec["true_size"]<1)])
fig_err_outbreak_case(resultRec.loc[(resultRec["true_remain"]/resultRec["true_size"]>=0) & (resultRec["true_remain"]/resultRec["true_size"]<1)])
