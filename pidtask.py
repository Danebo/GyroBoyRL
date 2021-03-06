"""
Code by Rasmus Loft and Andreas Danebo Jensen
"""

import multiprocessing
import time
import state
import generalFunctions

class PIDTask(object):

#gAng   - gyro angle in degrees
#gSpd   - gyro angle speed in degrees/sec
#mPos   - Rotation angle of motor in degrees
#mSpd   - Rotation speed of motor in degrees/sec
#mSum   - Sum of Positions of right and left tacho
#mD     - Delta tacho
#mDP1   - previous delta tacho
#mDP2   - previous previous delta tacho
#mDP3   - previous previous previous delta tacho
#pidPWR - motor power in [-100,100]
#cLo    - count numbers of balancing loops
#dt     - delta time since last balancing loop
#speed  - Default 0 for learning balance only

    debug = False
    CEspd = -0.01
    CEmSpd = 0.08
    CEmPos = 0.12
    CEgSPd = 0.8
    CEgAng = 15
    speed =0
    #get motor and gyro when executing task
    def __init__(self, state):
        self.state = state
    #get state from pidstate, and return new state
    def __call__(self):
        # self.state = state
        newState = self.getPIDPower()
        return newState
    def __str__(self):
        return 'Task %s,  %s' % (self.mSum, self.gSpd) # old shit

    #calculate PID output
    def getPIDPower(self):
        self.state.now = generalFunctions.getTime()
        if (self.state.cLo == 0):
            dt = 0.014
            self.state.start = generalFunctions.getTime()
        else:
            dt = ((self.state.now - self.state.start)/1000.0)/self.state.cLo
        self.state.cLo += 1
        
        if (self.debug):
            print("gSpd=",self.state.gSpd)
        self.state.gAng = self.state.gAng+(self.state.gSpd*dt) # integrate angle speed to get angle
        if (self.debug):
            print("gAng=")
            print(self.state.gAng)

        mD = self.state.mSum - self.state.mSumOld
        self.state.mSumOld = self.state.mSum
        self.state.mPos = self.state.mPos + mD
        self.state.mSpd = ((mD + self.state.mDP1 + self.state.mDP2 + self.state.mDP3) / 4.0) / dt # motor rotational speed
        self.state.mDP3 = self.state.mDP2
        self.state.mDP2 = self.state.mDP1
        self.state.mDP1 = mD

        # Compute new motor power
        self.state.mPos = self.state.mPos - self.speed * dt    # make GyroBoy go forward or backward
        pwr = self.CEspd * self.speed + self.CEmSpd * self.state.mSpd + self.CEmPos * self.state.mPos + self.CEgSPd * self.state.gSpd + self.CEgAng * self.state.gAng
        if (self.debug):
            print("pwr=")
            print(pwr)
        if (pwr > 100):
            pwr = 100
        if (pwr < -100):
            pwr = -100
        self.state.pwr = pwr
        return self.state    