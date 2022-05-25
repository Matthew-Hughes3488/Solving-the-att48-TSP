import math                        
import random                       
import pandas as pd                 
import numpy as np                  
import matplotlib.pyplot as plt     

np.set_printoptions(precision=4)
pd.set_option('display.max_rows', 20)
pd.set_option('expand_frame_repr', False)
pd.options.display.float_format = '{:,.2f}'.format

# 
def initParameter():
     
     #starting temprature
    tInitial = 3250
    #stop temprature         
    tFinal  = 0.1
    #Markov                  
    nMarkov = 10000 
    #T(k)=alfa*T(k-1)        
    alfa    = 0.98   

    return tInitial,tFinal,alfa,nMarkov

def read_TSPLib(fileName):

    res = []
    with open(fileName, 'r') as fid:
        for item in fid:
            if len(item.strip())!=0:
                res.append(item.split())

    loadData = np.array(res).astype('int')      
    coordinates = loadData[:,1::]
    return coordinates

def getDistMat(nCities, coordinates):

    distMat = np.zeros((nCities,nCities))       
    for i in range(nCities):
        for j in range(i,nCities):
            distMat[i][j] = distMat[j][i] = round(np.linalg.norm(coordinates[i]-coordinates[j]))
    return distMat                              

def calTourMileage(tourGiven, nCities, distMat):

    mileageTour = distMat[tourGiven[nCities-1], tourGiven[0]]   
    for i in range(nCities-1):                                  
        mileageTour += distMat[tourGiven[i], tourGiven[i+1]]
    return round(mileageTour)                     

def plot_tour(tour, value, coordinates):

    num = len(tour)
    x0, y0 = coordinates[tour[num - 1]]
    x1, y1 = coordinates[tour[0]]
    plt.scatter(int(x0), int(y0), s=15, c='r')     
    plt.plot([x1, x0], [y1, y0], c='b')             
    for i in range(num - 1):
        x0, y0 = coordinates[tour[i]]
        x1, y1 = coordinates[tour[i + 1]]
        plt.scatter(int(x0), int(y0), s=15, c='r')  
        plt.plot([x1, x0], [y1, y0], c='b')         

    plt.xlabel("Total mileage of the tour:{:.1f}".format(value))
    plt.title("Optimization tour of TSP{:d}".format(num))  
    plt.show()

def mutateSwap(tourGiven, nCities):

    i = np.random.randint(nCities)         
    while True:
        j = np.random.randint(nCities)      
        if i!=j: break                      

    tourSwap = tourGiven.copy()             
    tourSwap[i],tourSwap[j] = tourGiven[j],tourGiven[i] 

    return tourSwap

def main():

    fileName = "att48.txt"
    coordinates = read_TSPLib(fileName)    

    tInitial,tFinal,alfa,nMarkov = initParameter()

    nCities = coordinates.shape[0]
    distMat = getDistMat(nCities, coordinates)
    nMarkov = nCities
    tNow    = tInitial      

    tourNow   = np.arange(nCities)   
    valueNow  = calTourMileage(tourNow,nCities,distMat) 
    tourBest  = tourNow.copy()                          
    valueBest = valueNow                                
    recordBest = []                                     
    recordNow  = []                                     

    iter = 0                        
    while tNow >= tFinal:            

        for k in range(nMarkov): 
            
            tourNew = mutateSwap(tourNow, nCities)       
            valueNew = calTourMileage(tourNew,nCities,distMat) 
            deltaE = valueNew - valueNow

            if deltaE < 0:                          
                accept = True
                if valueNew < valueBest:             
                    tourBest[:] = tourNew[:]
                    valueBest = valueNew
            else:                                    
                pAccept = math.exp(-deltaE/tNow)     
                if pAccept > random.random():
                    accept = True
                else:
                    accept = False

            
            if accept == True:                      
                tourNow[:] = tourNew[:]
                valueNow = valueNew

        
        tourNow = np.roll(tourNow,2)                

        
        recordBest.append(valueBest)                 
        recordNow.append(valueNow)                   
        print('i:{}, t(i):{:.2f}, valueNow:{:.1f}, valueBest:{:.1f}'.format(iter,tNow,valueNow,valueBest))

        
        iter = iter + 1
        tNow = tNow * alfa                              

    figure1 = plt.figure()     
    plot_tour(tourBest, valueBest, coordinates)
    figure2 = plt.figure()     #  2
    plt.title("Optimization result of TSP{:d}".format(nCities)) 
    plt.plot(np.array(recordBest),'b-', label='Best')           
    plt.plot(np.array(recordNow),'g-', label='Now')             
    plt.xlabel("iter")                                          
    plt.ylabel("mileage of tour")                               
    plt.legend()                                                
    plt.show()

    print("Tour verification successful!")
    print("Best tour: \n", tourBest)
    print("Best value: {:.1f}".format(valueBest))

    exit()

if __name__ == '__main__':
    main()