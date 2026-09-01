# Physical-event methods

The analytical unit is a plot-level runoff event identified by:

`Year + Irrigation + Rep + Treatment`

Replicates remain distinct. `Date`, sample identifiers, chemistry analytes, and
measurement methods describe observations associated with an event but do not
create additional physical events. `EventDate` is the date of the unique
genuine volume observation when one is available; otherwise it is the earliest
valid contributing date.

Copied runoff-volume values repeated across chemistry rows contribute one
deterministically identified volume observation. Genuine parallel measurement
methods remain separate observations and are summarized according to the
configured resolution rule. Confirmed zero runoff is an observed zero; missing
runoff is not converted to zero.

The shared preflight in `results/preflight/` checks the event roster, dates,
event-level predictor consistency, copied volumes, multiple measurement
methods, missing observations, and blocking conflicts. The release contract is
528 plot-level events: 510 numeric-irrigation events and 18 events labeled S1
or S2.

For annual reporting, event-level loads are summed within
`Year + Treatment + Rep + Analyte + Draw`. Replicate-specific plot totals are
then averaged within treatment and year before uncertainty is summarized.
Observed replicate minimum-to-maximum ranges are descriptive and are not
confidence intervals.
