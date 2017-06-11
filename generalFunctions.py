"""
Code by Rasmus Loft and Andreas Danebo Jensen
"""

import time

def getTime():
        "NOAW"
        return int(round(time.time()*1000))

def scaleState(s):
    mPosS = 2000.0
    mSpdS = 2000.0
    gAngS = 15.0
    gSpdS = 100.0
    return [s[0]/mPosS,s[1]/mSpdS,s[2]/gAngS,s[3]/gSpdS]

def scaleAction(a):
    return a/100.0
    