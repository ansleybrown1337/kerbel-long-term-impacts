### Bayesian Model Results Overview

- Analyzed 15,000 data points across 27,000 parameters using Stan software
- Model runtime: 50 minutes, nearly 1GB of data
- Successfully captured tillage effects on water quality with honest uncertainty quantification
- Multi-output Gaussian process learns relationships between analytes across years
  - Analytes learn from each other (e.g., TSS and total phosphorus relationships)
  - Wider confidence bands in years with fewer data points

### Tillage Impact Rankings by Analyte

- Ranked by effect size from least to most affected by STIR values
- Highest impact: Total Suspended Solids (TSS) - narrowest confidence interval, strongest relationship
- Moderate impact: TKN, Total Phosphorus (TP), Total Soluble Phosphorus (TSP), Orthophosphate
- Pattern: Insoluble compounds less affected, soluble compounds more affected
- Negative coefficients indicate inverse relationships (more tillage = less of that analyte)

### Volume vs Concentration Effects

- Unexpected finding: More tillage correlates with less runoff volume
  - Potentially due to improved infiltration from conservation tillage
  - Could relate to soil cracking patterns over recent years
- Concentration model performs better than volume predictions
- Need to investigate volume relationship further, possibly split by time periods

### Data Quality and Measurement Challenges

- Multiple sampling methods create complexity: ISCO, low-cost samplers, grab samples (GB), grab hourly (GBH)
- TSS shows most bias between sampling methods
- Recommendation: Eliminate grab samples when multiple methods available in same year
- First ISCO sample purge issue affects TSS agreement - consider removing first samples

### Annual Load Estimates and Temporal Effects

- Model generates annual loads by simulating every irrigation event, then summing
- Captures 2-year temporal persistence for most analytes
- Strongest temporal effects: Orthophosphate, Total Phosphate, TSS, TSP
- Model performs well on analytes with strong STIR relationships (TP, TSS)
- Struggles with nitrate prediction (indicates tillage not strongly related to nitrate)

### Missing Variables and Model Improvements

- Need to add surface residue as key predictor
  - STIR affects residue, residue affects both concentration and volume
  - Use percent cover data (most available across years)
- Consider adding irrigation advance times (available 2011-2016)
  - Significant correlation with tillage and runoff patterns
  - Early years measured with stopwatch, later years more precise
- Explore adding treatment labels (CT/MT/ST) despite multicollinearity concerns

### Publication Strategy

- Primary paper: “STIR impacts on long-term water quality and legacy effects”
  - Focus on robust analytes: TSS, TP, TKN, orthophosphate
  - Eliminate analytes with <3 years of data
  - Include annual load tables for farmer/stakeholder communication
- Secondary paper: Bayesian vs machine learning comparison
  - Causal inference (Bayesian) vs prediction (ML) approaches
  - Commentary on appropriate use cases for each method
  - Potential collaboration with Leo on AI aspects

### Next Steps and Decisions

- Investigate negative volume-tillage relationship with theoretical backing
- Create operation table with STIR values, depths, frequencies for validation
- Determine final analyte list for publication (eliminate low-data analytes)
- Add residue cover and irrigation advance times to model
- Schedule meeting with Leo for machine learning collaboration discussion
- Collect penetrometer and moisture data for compaction assessment
- Make 2026 tillage treatment decisions based on current findings