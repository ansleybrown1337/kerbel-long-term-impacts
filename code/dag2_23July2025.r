dag {
bb="-5.892,-5.356,4.412,6.847"
"Conc. e" [latent,pos="-4.568,-3.064"]
"Crop Type" [exposure,pos="1.574,-0.546"]
"Depth measurement method" [exposure,pos="0.177,3.977"]
"Inflow Conc." [exposure,pos="-1.484,-3.787"]
"Inflow vol" [exposure,pos="-1.289,-0.605"]
"Irrigation Method" [exposure,pos="0.375,1.380"]
"Load e." [latent,pos="-5.102,0.612"]
"Outflow Conc." [outcome,pos="-3.089,-2.367"]
"Outflow vol" [outcome,pos="-1.647,2.641"]
"Sampler Method" [exposure,pos="-3.282,-4.554"]
"Soil Properties" [latent,pos="-0.305,-1.219"]
"Vol. e " [latent,pos="-2.522,4.234"]
"duplicate status" [exposure,pos="-3.968,-3.951"]
"flume type" [exposure,pos="-2.005,-0.091"]
Analyte [exposure,pos="-2.626,-4.327"]
Block [exposure,pos="-3.104,0.582"]
Load [outcome,pos="-4.515,2.568"]
STIR [exposure,pos="0.252,-3.050"]
"Conc. e" -> "Outflow Conc."
"Crop Type" -> "Soil Properties"
"Depth measurement method" -> "Outflow vol"
"Inflow Conc." -> "Outflow Conc."
"Inflow Conc." -> "Soil Properties"
"Inflow vol" -> "Outflow Conc."
"Inflow vol" -> "Outflow vol"
"Inflow vol" -> "Soil Properties"
"Irrigation Method" -> "Inflow vol"
"Load e." -> Load
"Outflow Conc." -> Load
"Outflow vol" -> Load
"Sampler Method" -> "Outflow Conc."
"Soil Properties" -> "Outflow Conc."
"Soil Properties" -> "Outflow vol"
"Vol. e " -> "Outflow vol"
"duplicate status" -> "Outflow Conc."
"flume type" -> "Outflow Conc."
"flume type" -> "Outflow vol"
Analyte -> "Outflow Conc."
Block -> "Outflow Conc."
Block -> "Outflow vol"
STIR -> "Soil Properties"
}
