dag {
bb="-5.892,-5.356,4.412,6.847"
"Conc. e" [latent,pos="-4.568,-3.064"]
"Crop Type" [exposure,pos="1.574,-0.546"]
"Inflow Conc." [pos="-2.696,0.800"]
"Inflow vol" [exposure,pos="-1.573,-0.427"]
"Load e." [latent,pos="-4.541,4.970"]
"Measurement method" [exposure,pos="0.177,3.977"]
"Outflow Conc." [outcome,pos="-3.089,-2.367"]
"Outflow vol" [outcome,pos="-1.647,2.641"]
"Sampler Method" [exposure,pos="-2.503,-4.339"]
"Soil Properties" [latent,pos="-0.305,-1.219"]
"Trt " [exposure,pos="0.252,-3.050"]
"Vol. e " [latent,pos="-2.681,4.749"]
Analyte [exposure,pos="-2.293,-0.497"]
Block [pos="-3.596,1.414"]
Load [outcome,pos="-4.515,2.568"]
"Conc. e" -> "Outflow Conc."
"Crop Type" -> "Soil Properties"
"Inflow Conc." -> "Outflow Conc."
"Inflow vol" -> "Inflow Conc."
"Inflow vol" -> "Outflow Conc."
"Inflow vol" -> "Outflow vol"
"Inflow vol" -> "Soil Properties"
"Load e." -> Load
"Measurement method" -> "Outflow vol"
"Outflow Conc." -> Load
"Outflow vol" -> Load
"Sampler Method" -> "Outflow Conc."
"Soil Properties" -> "Outflow Conc."
"Soil Properties" -> "Outflow vol"
"Trt " -> "Soil Properties"
"Vol. e " -> "Outflow vol"
Analyte -> "Inflow Conc."
Analyte -> "Outflow Conc."
Block -> "Outflow Conc."
Block -> "Outflow vol"
}
