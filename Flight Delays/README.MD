# Delayed Flights Prediction

<img src = 'https://img.jakpost.net/c/2018/08/07/2018_08_07_50869_1533633650._large.jpg'
     width = "550"/>

# 1. Problem Background and Motivation <a class="anchor" id="problem-background"></a>

>- High level outline
>- Brief background section
>>- What is the problem?
>>- Why is it important?
>>- Who are the key stakeholders?

<b>High level outline:</b>
This is a prediction model for predicting delays in flights for a commercial airline.

<b>Brief background section</b>
scheduled_departure_time, carrier, destination, distance,
               origin, weather, day
               
The dataset has 2200 datapoints including the features as mentioned below:
- Scheduled departure time
- Carrier
- Destination
- Distance
- Origin
- Weather
- Day

> <b>What is the problem?</b><br>
This problem deals with predicting the delays in scheduled flights for a commercial airline. Prediction of such delays will ensure that there are no operational issues and will help streamline the flight scheduling.

> <b>Why is it important?</b><br>
As people are increasingly choosing to travel by air and with greater frequency, the number of flights that fail to take off on time has also increased. This increase in delays has worsened the crowded situation at airports and has caused financial difficulties within the airline industry. Air transportation delay indicates the lack of efficiency in the aviation system. It is a high cost to both airline companies and their passengers. According to the estimation by the Total Delay Impact Study, the total cost of air transportation delay to air travelers and the airline industry in 2007 was \\$32.9 billion in the US, resulting in a \\$4 billion reduction in GDP. The chart below shows how the cost per minute of delay is increasing every year. 

<br>
<img src="http://wp-uploads-trefis.s3.amazonaws.com/articles/wp-content/uploads/2016/08/tab.png"
     width = "700"/>
<br>
From this chart, we can see that the per minute cost of delay is majorly on account of Fuel and Oil, Crew, and Maintenance.
<br>

Delays affect both passengers and air carriers. Increased travel time and increased expenses on food and lodging cause stress among passengers. Further, delays disrupt the purpose of air travel — rapid, affordable and safe — and make the passengers distrust airlines. On the other hand, airlines suffer from extra crew costs, costs associated with accommodating disrupted passengers, and aircraft re-positioning, as airline fleet and crew schedules are largely based on the scheduled times.

In addition to the costs borne by passengers and airlines, there may be an indirect circular impact on the rest of the economy. As airline costs increase, they charge passengers a higher fare, affecting leisure travels and those industries which rely on air transportation for their business. This indirect impact then results in spillover on the rest of the economy, snowballing the already high costs of airlines.
Therefore, predicting flight delays can improve airline operations and passenger satisfaction, which will result in a positive impact on the economy. 

Delay of the flight can be dependent upon various features like shceduled departure time, carrier information, weather condition etc. We can use dataset containing these features to predict the flight delays.
<br>
<br>
<img src="https://d3i71xaburhd42.cloudfront.net/29e2a5a6b72d6738c6feb41ee0f8a9b57f600e7d/7-Figure3-1.png"
     width = "550"/>

<img src="http://scientistseeksdata.weebly.com/uploads/4/0/1/1/40118629/7404805_orig.png"
     width = "700"/>
     
From this, we can see that in the United States, on average, around 20% of the flights get delayed. Based on the high number of flights that run on a daily basis, 20% amounts to a very large number of delayed flights. 
Also, the reasons for the delay is as mentioned below:

|          Reason            |      %      |
|----------------------------|-------------|
|    Late aircraft delay     |    38.2%    |
|       NAS delay            |    29.5%    |
|      Carrier delay         |    28.9%    |
|      Weather delay         |    3.20%    |
|      Security delay        |    0.20%    |

> <b>Who are the key stakeholders?</b><br>
Accurately predicting flight delays will help commercial airlines and passengers (indirectly). Airlines will be able to anticipate and plan for predicted delays. Passengers also stand to indirectly benefit from such prediction, as they will not have to waste time at the airport, but can instead come to the airport directly in time for the flight.
