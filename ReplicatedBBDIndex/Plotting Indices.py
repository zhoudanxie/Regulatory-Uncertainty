# Author: Zhoudan Xie
# Date: May 12, 2020

# Import packages
import pandas as pd
import os
import re
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

#----------------------------------------------Plot the Indices--------------------------------------------------------

# Data
index=pd.read_csv('Replicated Uncertainty Indices.csv')
index['date']=index['year-month'].astype('datetime64[ns]').dt.date

#-----------------------------------------------------------------------------------------------------------------------
# Plot EPU & PU
x=index['date']
y1=index['EPU']
y2=index['PU']

years = mdates.YearLocator(2)  # every two years
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

fig, ax = plt.subplots(1, figsize=(13,9))
fig.subplots_adjust(top=0.92,bottom=0.02,left=0.08,right=0.98)
ax.plot(x,y1,color='#033C5A',label="Economic Policy Uncertainty",linewidth=1.5)
ax.plot(x,y2,color='#AA9868',label="Policy Uncertainty",linewidth=1.5)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(index['date'].iloc[0], 'Y')
datemax = np.datetime64(index['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
ax.grid(False)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14)
ax.set_ylabel('Uncertainty Index',fontsize=24)
ax.set_yticks(np.arange(0,max(max(y1),max(y2))+100,100))
ax.grid(color='gray', which='major', axis='y', linestyle='dashed')

# Legend and title
fig.legend(loc=9,bbox_to_anchor=(.35, .41, .6, .5),fontsize=14)
ax.set_title('(Based on four major U.S. newspapers)',fontsize=18)
fig.suptitle('Figure 1: U.S. Monthly Policy Uncertainty Index, 1985-2020',
                y=0.98,fontsize=22)

# Inset plot
xins=x.iloc[-6:]
y1ins=y1.iloc[-6:]
y2ins=y2.iloc[-6:]

axins=inset_axes(ax, width=5, height=3, bbox_to_anchor=(.04, .48, .6, .5),
                    bbox_transform=ax.transAxes,loc=2)

axins.plot(xins,y1ins,color='#033C5A',linewidth=2,marker='D',markersize=8)
axins.plot(xins,y2ins,color='#AA9868',linewidth=2,marker='D',markersize=8)
axins.format_xdata = mdates.DateFormatter('%Y-%m')
axins.set_yticks(np.arange(150,max(max(y1ins),max(y2ins))+100,100))
axins.grid(color='gray', which='major', axis='y', linestyle='dotted')
axins.tick_params(axis='both',which='major',labelsize=14)
axins.set_facecolor('#d3d3d3')
axins.set_alpha(0.5)
axins.set_title('Uncertainty over the Past Six Months',fontsize=16,position=(0.5,0.9))
#mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

# Notes
fig.text(0.065,0.07,"Notes: Indices are calculated and plotted by the author applying the Baker, Bloom, and Davis (2016) "
                  "method to four U.S. newspapers\nincluding Chicago Tribune, Los Angeles Times, New York Times, "
                  "and The Washington Post.\nEach index is normalized seperately to mean 100 from January 1985 through December 2009.",
         fontsize=16,style='italic')

plt.savefig('EPU&PU.png')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Plot PU & RPU
x=index['date']
y1=index['RPU']
y2=index['PU']

years = mdates.YearLocator(2)   # every two years
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

fig, ax = plt.subplots(1, figsize=(13,9))
fig.subplots_adjust(top=0.92,bottom=0.02,left=0.08,right=0.98)

ax.plot(x,y1,color='#033C5A',label="Regulatory Policy Uncertainty",linewidth=1.5)
ax.plot(x,y2,color='#AA9868',label="Policy Uncertainty",linewidth=1.5)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(index['date'].iloc[0], 'Y')
datemax = np.datetime64(index['date'].iloc[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x
ax.grid(False)
ax.set_title('Figure 2: U.S. Monthly Policy Uncertainty Index, 1985-2020',fontsize=22)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

# Set tick and label format
ax.tick_params(axis='both',which='major',labelsize=14)
ax.set_ylabel('Uncertainty Index',fontsize=24)
ax.set_yticks(np.arange(0,max(max(y1),max(y2))+100,100))
ax.grid(color='gray', which='major', axis='y', linestyle='dashed')

# Legend
fig.legend(loc=9,bbox_to_anchor=(.35, .41, .6, .5),fontsize=14)
fig.suptitle('Figure 2: U.S. Monthly Regulatory Policy Uncertainty Index, 1985-2020',
                y=0.98,fontsize=22)
ax.set_title('(Based on four major U.S. newspapers)',fontsize=18)

# Inset plot
xins=x.iloc[-6:]
y1ins=y1.iloc[-6:]
y2ins=y2.iloc[-6:]

axins=inset_axes(ax, width=5, height=3, bbox_to_anchor=(.04, .48, .6, .5),
                    bbox_transform=ax.transAxes,loc=2)

axins.plot(xins,y1ins,color='#033C5A',linewidth=2,marker='D',markersize=8)
axins.plot(xins,y2ins,color='#AA9868',linewidth=2,marker='D',markersize=8)
axins.format_xdata = mdates.DateFormatter('%Y-%m')
axins.set_yticks(np.arange(50,max(max(y1ins),max(y2ins))+100,100))
axins.grid(color='gray', which='major', axis='y', linestyle='dotted')
axins.tick_params(axis='both',which='major',labelsize=14)
axins.set_facecolor('#d3d3d3')
axins.set_alpha(0.5)
axins.set_title('Uncertainty over the Past Six Months',fontsize=16,position=(0.5,0.9))
#mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

# Notes
fig.text(0.065,0.07,"Notes: Indices are calculated and plotted by the author applying the Baker, Bloom, and Davis (2016) "
                  "method to four U.S. newspapers\nincluding Chicago Tribune, Los Angeles Times, New York Times, "
                  "and The Washington Post.\nEach index is normalized seperately to mean 100 from January 1985 through December 2009.",
         fontsize=16,style='italic')

plt.savefig('RPU&PU.png')
plt.show()