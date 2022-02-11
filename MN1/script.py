import matplotlib.pyplot as plt
import pandas as pd

def base(N): #calculates (1- alpha)
    return 1 - (2 / (N + 1))

def EMA(N, tab): #tab [p0, p1, p2, ... , pn]
    up = tab[0]
    down = 1
    b = base(N)
    for x in range(1,N+1):#<1,N>
        up = up + (b ** x) * tab[x]
        down = down + b ** x
    return up/down

data = pd.read_csv('btc.csv')
macdData = [0] * 1100
signalData = [0] * 1100

dates = data['Timestamp'].values
prices = data['market-price'].values

for x in range(27, 36): #<27, 35> #calculate start macd values
    dataSubset = prices[x-27:x] #data from oldest to today
    dataSubset = dataSubset[::-1]
    ema12 = EMA(12, dataSubset[0:13])
    ema26 = EMA(26,dataSubset)
    macd = ema12 - ema26
    macdData[x - 27] = macd #macd[0] - macd[8]
# i = 0
for x in range(36,1100): #<35, 1034>
    dataSubset = prices[x-35:x] #data from oldest to today
    dataSubset = dataSubset[::-1] #data from today to oldest #reverse subset to pass to ema
    ema12 = EMA(12, dataSubset[0:13])
    ema26 = EMA(26,dataSubset[0:27])
    macd = ema12 - ema26
    macdData[x - 27] = macd #macd[9] - macd[999]

    sample = macdData[x-27-9:x-26]
    sample = sample[::-1]
    signal = EMA(9,sample[0:10])
    signalData[x-36] = signal #signalData[0] - signalData[990]

    # i = i + 1
    # print(i)

macdData = macdData[9:1009] #1000 macd
signalData = signalData[0:1000]
dates = dates[35:1035]
prices = prices[35:1035]
numbers = list(range(1,1001))

plt.plot(numbers, macdData)
plt.plot(numbers, signalData)

plt.plot(numbers, prices)
plt.legend(['macd','signal','price'])
plt.show()

#decision making

#linia MACD przecina linię sygnału od dołu – jest to sygnał do zakupu akcji i zapowiedź trendu wzrostowego.
#linia MACD przecina linię sygnału od góry – jest to sygnał do sprzedaży akcji i zapowiedź odwrócenia trendu. #2

account = 1000 #$1000
ownedBtc = 0
print("Starting with $" + str(account))
for t in range(1,1000): #1 - skip first moment
    if macdData[t] > signalData[t] and macdData[t-1] < signalData[t-1] and account > 0: #1
        #buy spend 50% of account
        budget = account / 2
        newBtc = budget / prices[t]

        account = account - budget
        ownedBtc = ownedBtc + newBtc
        # print("Buy " + str(t))
    elif macdData[t] < signalData[t] and macdData[t-1] > signalData[t-1] and ownedBtc > 0: #2
        #sell everything of btc
        newAccount = ownedBtc * prices[t]
        ownedBtc = 0
        account = account + newAccount
        # print("Sell " + str(t))
print("Ending with $" + str(round(account,2)))