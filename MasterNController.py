"""
Code by Rasmus Loft and Andreas Danebo Jensen
"""
import multiprocessing
import time
import pidstate
import pidtask
import slstate
import sltask
import generalFunctions
import time
import brickpi3
import sys
import state
import copy
import ddpg
import os
#import dummy
import atexit
from threading import Condition
from threading import Thread


#class master ():
BP=brickpi3.BrickPi3()


#Ports
PORT_SENSOR_GYRO = BP.PORT_2
PORT_MOTOR_RIGHT = BP.PORT_A
PORT_MOTOR_LEFT  = BP.PORT_B


#Constants

# Variables
gOS = 0 # gyro offset
power100 = 0 # start time of power set to max
pidInControl = True
SLMaxCount = 3
SLcounter =0
SLiterations = 0

loopCount =0
executionStart = generalFunctions.getTime()
exiting = False
Logging = False

# Establish communication queues
queuePID = multiprocessing.Queue()
queueSL = multiprocessing.Queue()	
stateQueue = multiprocessing.Queue()	


#experience = (s, a, r, t, s2)
experienceQueue = multiprocessing.Queue()

# self learning system output network
networkQueue  = multiprocessing.Queue()

#DDPG training cycle done:
dDPGTrainDone = multiprocessing.Queue()
dDPGTrainStart = multiprocessing.Queue()

gAngList = [] #List to store gAng readings for logging
gSpdList = [] #List to store gSpd readings for logging
dampenedSumOfSpd = 0.0


#constants
dampFactor = 0.96 # dampens the gSpg
loopDelay = 0.008 # delay between loops, should not go under 3 ms for sensors.

startTrainingLim = 0.6# lower means waiting longer for stability
stopTrainingLim = 1.4 # higher means trusting the PID controller to recover more extreme situations.


forwardEstimate = 3  # number of cycels that the estimator looks ahead

trainingSeconds = 60
 
# define functions

def getGyroDPS():
    try:
        if (Logging == True):
            return BP.get_sensor(PORT_SENSOR_GYRO)[1]
        else:
            return BP.get_sensor(PORT_SENSOR_GYRO)
    except:
        print('gyro failed, re-initializing')
        initGYRO()
        return getGyroDPS()
  
def getGyroAPSDPS():
    # if (Logging == True):
    reading = BP.get_sensor(PORT_SENSOR_GYRO)
    return (reading[0],reading[1])
    # else:
        # return BP.get_sensor(PORT_SENSOR_GYRO)
  
  
def SafeExit():
    global exiting
    if (exiting==False):
        exiting = True
        print("safe exit")
        print("execution time was: {} seconds".format((generalFunctions.getTime()-executionStart)/1000))
        BP.reset_all()
        #shut down worker process
        queuePID.put(None)  
        experienceQueue.put(None)
        queueSL.put(None)
        time.sleep(5)
        if (Logging == True):
            log = open('log.csv','w')
            log.write("gAng, gSpd\n")
            gAngList.reverse()
            gSpdList.reverse()
            while (len(gAngList) > 0):
                log.write("%d,%d\n" % (gAngList.pop(), gSpdList.pop()))
            log.close()
        sys.exit()
    
def exit_handler():
    print('exit handeler running')
    if (exiting==False):
        print('program aborted from console')
        SafeExit()

atexit.register(exit_handler)
    
def initGYRO():
    time.sleep(0.25)
    Continue = False
    while not Continue:
        try:
            reading = BP.get_sensor(PORT_SENSOR_GYRO)
            Continue = True
        except brickpi3.SensorError:
            pass
        time.sleep(0.1)

def resetMotors():
    # Reset Motor Position
    BP.set_motor_power(PORT_MOTOR_LEFT,0)
    BP.set_motor_power(PORT_MOTOR_RIGHT,0)
    
    time.sleep(0.2)
    BP.offset_motor_encoder(PORT_MOTOR_LEFT, BP.get_motor_encoder(PORT_MOTOR_LEFT))
    BP.offset_motor_encoder(PORT_MOTOR_RIGHT, BP.get_motor_encoder(PORT_MOTOR_RIGHT))


        
# set ports
if(Logging==True):
    BP.set_sensor_type(PORT_SENSOR_GYRO,BP.SENSOR_TYPE.EV3_GYRO_ABS_DPS)
else:
    BP.set_sensor_type(PORT_SENSOR_GYRO,BP.SENSOR_TYPE.EV3_GYRO_DPS)


# make sure voltage is high enough
if BP.get_voltage_battery() < 7:
    print("Battery voltage below 7v, too low to run motors reliably. Exiting.")
    SafeExit()

# Reset motor position and speed    
resetMotors()    
    
# gyro initialization to avoid fail
initGYRO()

executionStart = generalFunctions.getTime()

# start DDPG
ddpg = ddpg.ddpgClass(networkQueue, experienceQueue,dDPGTrainDone,dDPGTrainStart,queueSL,trainingSeconds)
ddpg.start()


# Start consumers
pidState = pidstate.PIDState(queuePID, stateQueue)
pidState.start()

slState = slstate.SLState(queueSL, stateQueue,networkQueue)
slState.start()

#run at high priority
os.nice(-20)


while (True):

    initGYRO()
    
    #reset vars:
    dampenedSumOfSpd = 0.0
    power100 = generalFunctions.getTime()
    pidInControl = True
    SLiterations = 0
    #gOS = orgGOS

    #calibrate gyroscope:
    gMax = 1000
    gMin = -1000
    timeout =0
    BP.set_led(0)
    while ((gMax-gMin) > 2):
        gMax = -1000
        gMin = 1000
        gSum = 0
        for i in range(0,200):
            gyroRate = getGyroDPS()
            gSum += gyroRate
            if (gyroRate > gMax):
                gMax = gyroRate
            if (gyroRate < gMin):
                gMin = gyroRate
            time.sleep(0.004)
        gOS = gSum / 200.0
        orgGOS = gOS
        print("calibration")
        print(gOS)
        if (timeout==10):
            SafeExit()
        timeout=timeout+1
        BP.set_led(100)
    

    #start state reset:
    CurState = state.State()
    CurState.cLo=0
    CurState.gAng = -0.25
    CurState.mSumOld = 0
    CurState.mPos = 0
    CurState.mDP1 = 0
    CurState.mDP2 = 0
    CurState.mDP3 = 0

    sOld = None
    oldTime = generalFunctions.getTime()

    print('Ready, remove pedestal')
    while (True):
            loopStart = generalFunctions.getTime();
            #Add latest reading to Arraynif logging
            if(Logging==True):
                gAng,gSpd = getGyroAPSDPS()
                gSpdList.append(gSpd)
                gAngList.append(gAng)
            else:
                gSpd= getGyroDPS()
            
            gOS = 0.0005*gSpd+(1.0-0.0005)*gOS # update offset
            gSpd = gSpd - gOS # remove offset
            #estimation = dampenedSumOfSpd*dampFactor+forwardEstimate*gSpd*loopDelay
            #dampenedSumOfSpd = dampenedSumOfSpd*dampFactor+gSpd*loopDelay
            
            

            #consider saving gspd after this point.
            rightTacho = BP.get_motor_encoder(PORT_MOTOR_RIGHT)
            leftTacho = BP.get_motor_encoder(PORT_MOTOR_LEFT)
            mSum = rightTacho + leftTacho
                      
            CurState.mSum = mSum
            CurState.gSpd = gSpd
            if (pidInControl == True):
                queuePID.put(pidtask.PIDTask(CurState)) # calculate PID output in seperate process
            else:
                
                queueSL.put(sltask.SLTask(CurState))
                #print('sl task put in queue')
            
            # Blocking
            CurState = stateQueue.get(timeout=3)
            
            estimation = CurState.gAng + forwardEstimate*CurState.gSpd*loopDelay
                        
            dynamicDelay = (loopDelay*1000-(generalFunctions.getTime()-loopStart))/1000.0
            if(dynamicDelay<0):
                dynamicDelay=0
            time.sleep(dynamicDelay) #dynamic wait
            
            pidPWR = CurState.pwr
            
            # update experiance Replay queue
            terminate = False
            if(sOld is not None): # run from second timestep
                #calculate reward
                scaledState = generalFunctions.scaleState([CurState.mPos,CurState.mSpd,CurState.gAng,CurState.gSpd])
                x = scaledState[0]
                theta = scaledState[2]
                theta_dot = scaledState[3]
                action = generalFunctions.scaleAction(aOld)
                punish =  -(pow(theta,2) + 0.1*pow(theta_dot,2) + 0.001*pow(action,2)+ 0.1*pow(x,2))
                reward = 2+punish
                experience = (sOld, aOld, reward, terminate, [CurState.mPos,CurState.mSpd,CurState.gAng,CurState.gSpd])
                experienceQueue.put(experience)
                
                if (pidInControl == False):
                    for _ in range(50):
                        experienceQueue.put(experience)


            # save current state as previous state for next time step
            sOld = [CurState.mPos,CurState.mSpd,CurState.gAng,CurState.gSpd]
            aOld = pidPWR
            
            #print('setting motor spd')
            BP.set_motor_power(PORT_MOTOR_LEFT,pidPWR)
            BP.set_motor_power(PORT_MOTOR_RIGHT,pidPWR)
          
            
            # stop robot if max power output for 1 second.
            if(abs(pidPWR)<100):
                power100 = generalFunctions.getTime()            
            
            if(generalFunctions.getTime() - power100 > 1000):
                print('power 100 over 1 sec') 
                break # more than 1 second at 100 power
            
            
            loopCount=loopCount+1
            pidTime = 1 #seconds
            #transfer control of robot
            if ((not dDPGTrainDone.empty()) and abs(estimation) < startTrainingLim and pidInControl == True and loopCount > (pidTime/loopDelay)):
                
                print("estimation before SL: %.2f" % estimation)
                
                SLcounter = SLcounter + 1
                
                if (SLcounter % SLMaxCount == 0):
                    dDPGTrainDone.get()
                    dDPGTrainStart.put(1)
                    print("Self learning in control: ",SLcounter/SLMaxCount)
                pidInControl = False
                BP.set_led(0)
            #retake control of robot
            #check if previous training is done
            if(abs(estimation) > stopTrainingLim and pidInControl == False): #175 ok! # 200 ok # 225 ok(fail at MAXpower)
                print("SL iterations: ", SLiterations)
                SLiterations = 0
                print("estimation after SL: %.2f" % estimation)
                print("PID IS IN THE HOUSE")
                pidInControl = True
                loopCount=0           
                BP.set_led(100)
                
            if (pidInControl == False):
                SLiterations = SLiterations + 1
            oldTime = generalFunctions.getTime()
    
    #wait for gyro stable arround 0 again
    #100 readings from gyro, avg, if abs smaller than 2
    
    print('Fallen')
    resetMotors()
    time.sleep(10.0)
    angleSpeedavg = 100
    upright = True
    while(upright):
        if(Logging==True):
            _,angleSpeed = getGyroAPSDPS()
        else:
            angleSpeed= getGyroDPS()
        angleSpeedavg = angleSpeedavg *0.999 + 0.001 * abs(angleSpeed)
        if (abs(angleSpeedavg)<0.5):
            print('I got up!')
            upright = False