Uses flask,ngrok and ML to predict the yield of the land based on parameters.

Use the following command to get the yield value

curl -H "Content-Type:application/json" -X POST -d '{"Latitude":46.9298,"Longitude":-118.352,"apparentTemperatureMax":18.61,"apparentTemperatureMin":-3.01,"cloudCover":0,"dewPoint":6.77,"humidity":0.69,"precipIntensityMax":0,"precipAccumulation":0,"precipTypeIsRain":0,"precipTypeIsSnow":0,"pressure":1027.95,"temperatureMax":23.93,"temperatureMin":6.96,"windBearing":9,"windSpeed":3.8,"NDVI":136.18}'  http://eb2393e9.ngrok.io

change the link to the url you receive from ngrok
