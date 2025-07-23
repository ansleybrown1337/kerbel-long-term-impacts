dag {
bb="-5.892,-5.356,4.412,6.847"
"Conc. e" [latent,pos="-4.568,-3.064"]
"Crop Type" [exposure,pos="1.574,-0.546"]
"Inflow Conc." [pos="-2.949,1.117"]
"Inflow vol" [exposure,pos="-1.190,0.691"]
"Load e." [latent,pos="-4.541,4.970"]
"Measurement method" [exposure,pos="0.177,3.977"]
"Outflow Conc." [outcome,pos="-3.004,-2.337"]
"Outflow vol" [outcome,pos="-1.647,2.641"]
"Sampler Method" [exposure,pos="-2.503,-4.339"]
"Soil Properties" [latent,pos="-0.226,-0.566"]
"Trt " [exposure,pos="0.883,-2.496"]
"Vol. e " [latent,pos="-2.681,4.749"]
Analyte [exposure,pos="-2.298,-0.111"]
Block [pos="-3.606,2.017"]
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

