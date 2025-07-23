# Directed Acyclic Graphs (DAGs) in R
# dagitty.net/dags.html

# TODO:
Consider adding fertilizer application quantities to the DAG
consider adding soil residue at planting
consider adding soil moisture at planting
Consider adding fertilizer application method

dag {
    bb="-3.052,-3.426,2.385,2.376"
    "Crop Type" [exposure,pos="-2.509,-1.165"]
    "Observed Concentration" [outcome,pos="0.594,-2.211"]
    "Other Uncertainty" [latent,pos="1.994,-3.017"]
    "Sampler Method" [exposure,pos="1.543,0.553"]
    "Soil Bulk Density" [exposure,pos="-2.039,0.571"]
    "Soil CEC" [exposure,pos="0.795,0.606"]
    "Soil EC" [exposure,pos="-0.017,0.845"]
    "Soil NO3-N" [exposure,pos="-1.314,0.886"]
    "Soil NaHCO3 P" [exposure,pos="-0.506,0.898"]
    "Soil OM%" [exposure,pos="-2.380,-0.353"]
    "True Concentration" [latent,pos="1.531,-2.187"]
    Analyte [exposure,pos="-1.314,-2.766"]
    Block [exposure,pos="-2.278,-2.742"]
    Treatment [exposure,pos="-2.468,-2.012"]
    "Crop Type" -> "Observed Concentration"
    "Crop Type" -> Analyte
    "Observed Concentration" -> "True Concentration"
    "Other Uncertainty" -> "True Concentration"
    "Sampler Method" -> "Observed Concentration"
    "Sampler Method" -> Analyte
    "Soil Bulk Density" -> "Observed Concentration"
    "Soil Bulk Density" -> Analyte
    "Soil CEC" -> "Observed Concentration"
    "Soil CEC" -> Analyte
    "Soil EC" -> "Observed Concentration"
    "Soil EC" -> Analyte
    "Soil NO3-N" -> "Observed Concentration"
    "Soil NO3-N" -> Analyte
    "Soil NaHCO3 P" -> "Observed Concentration"
    "Soil NaHCO3 P" -> Analyte
    "Soil OM%" -> "Observed Concentration"
    "Soil OM%" -> Analyte
    Analyte -> "Observed Concentration"
    Block -> "Observed Concentration"
    Block -> Analyte
    Treatment -> "Observed Concentration"
    Treatment -> Analyte
}
