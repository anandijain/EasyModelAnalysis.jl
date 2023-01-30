# Scenario 3: Limiting Deaths

Load packages:

```@example scenario3
using EasyModelAnalysis, AlgebraicPetri, UnPack
```

## Generate the Model and Dataset

```@example scenario3
function formSEIRHD()
    seirhd = LabelledPetriNet(
        [:S, :E, :I, :R, :H, :D],
        :exp => ([:S, :I] => [:E, :I]),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :hosp => (:I => :H),
        :death => (:H => :D)
    )
end

seirhd = formSEIRHD()
sys_seirhd = sys1 = ODESystem(seirhd)
```

```@example scenario3
function formSEIRD()
    SEIRD = LabelledPetriNet([:S, :E, :I, :R, :D],
	  :exp => ((:S, :I)=>(:E, :I)),
	  :conv => (:E=>:I),
	  :rec => (:I=>:R),
	  :death => (:I=>:D),
	)
    return SEIRD
end

seird = formSEIRD()
sys_seird = sys2 = ODESystem(seird)
```

```@example scenario3
function formSIRHD()
    SIRHD = LabelledPetriNet([:S, :I, :R, :H, :D],
	  :exp => ((:S, :I)=>(:I, :I)),
	  :rec => (:I=>:R),
	  :hosp => (:I=>:H),
      :death => (:H=>:D),
	)
    return SIRHD
end

sirhd = formSIRHD()
sys_sirhd = sys3 = ODESystem(sirhd)
```

```@example scenario3
function form_seird_renew()
    seird_renew = LabelledPetriNet([:S, :E, :I, :R, :D],
	  :exp => ((:S, :I)=>(:E, :I)),
	  :conv => (:E=>:I),
	  :rec => (:I=>:R),
	  :death => (:I=>:D),
      :renew => (:R=>:S)
	)
    return seird_renew
end

seird_renew = form_seird_renew()
sys_renew = sys4 = ODESystem(seird_renew)
```

```@example scenario3
function form_seird_detect()
    seirhd_detect = LabelledPetriNet([:S, :E, :I, :R, :H, :D],
        :exp => ((:S, :I) => (:E, :I)),
        :conv => (:E => :I),
        :rec => (:I => :R),
        :ideath => (:I => :D),
        :hosp => (:I => :H), # this is detection rate
        :hrec => (:H => :R), 
        :death => (:H => :D))
end
seirhd_detect = form_seird_detect()
sys_detect = sys5 = ODESystem(seirhd_detect)
```

```julia
using ASKEM # Hack, remove when merged
max_e_h = mca(seird, sirhd)
AlgebraicPetri.Graph(max_e_h[1])
```

```julia
max_3way = mca(max_e_h[1], seirhd)
AlgebraicPetri.Graph(max_3way[1])
```

```julia
max_seird_renew = mca(seird, seird_renew)
AlgebraicPetri.Graph(max_seird_renew[1])
```

```@example scenario3
t = ModelingToolkit.get_iv(sys1)
@unpack S, E, I, R, H, D = sys1
@unpack exp, conv, rec, hosp, death = sys_seirhd
NN = 10.0
@parameters u_expo=0.2 * NN u_conv=0.2 * NN u_rec=0.8 * NN u_hosp=0.2 * NN u_death=0.1 * NN N=NN
translate_params = [exp => u_expo / N,
    conv => u_conv / N,
    rec => u_rec / N,
    hosp => u_hosp / N,
    death => u_death / N]
subed_sys = substitute(sys1, translate_params)
sys = add_accumulations(subed_sys, [I])
@unpack accumulation_I = sys
```

```@example scenario3
u0init = [
    S => 0.9 * NN,
    E => 0.05 * NN,
    I => 0.01 * NN,
    R => 0.02 * NN,
    H => 0.01 * NN,
    D => 0.01 * NN,
]

tend = 6 * 7
ts = 0:tend
prob = ODEProblem(sys, u0init, (0.0, tend))
sol = solve(prob)
plot(sol)
```

## Model Analysis

### Question 1

> Provide a forecast of cumulative Covid-19 cases and deaths over the 6-week period from May 1 – June 15, 2020 under no interventions, including 90% prediction intervals in your forecasts. Compare the accuracy of the forecasts with true data over the six-week timespan.

```@example scenario3
get_uncertainty_forecast(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)
```

```@example scenario3
plot_uncertainty_forecast(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)], 6 * 7)
```

```@example scenario3
get_uncertainty_forecast_quantiles(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)],
                                   6 * 7)
```

```@example scenario3
plot_uncertainty_forecast_quantiles(prob, accumulation_I, ts, [u_conv => Uniform(0.0, 1.0)],
                                    6 * 7)
```

### Question 2

> Based on the forecasts, do we need additional interventions to keep cumulative Covid deaths under 6000 total? Provide a probability that the cumulative number of Covid deaths will stay under 6000 for the next 6 weeks without any additional interventions.

```@example scenario3
_prob = remake(prob, tspan = (0.0, 6 * 7.0))
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN]) # TODO: explain 0.4*NN
```

### Question 3

> We are interested in determining how effective it would be to institute a mandatory mask mandate for the duration of the next six weeks. What is the probability of staying below 6000 cumulative deaths if we institute an indefinite mask mandate starting May 1, 2020?

```@example scenario3
_prob = remake(_prob, p = [u_expo => 0.02])
prob_violating_threshold(_prob, [u_conv => Uniform(0.0, 1.0)], [accumulation_I > 0.4 * NN])
```

### Question 4

> We are interested in determining how detection rate can affect the accuracy and uncertainty in our forecasts. In particular, suppose we can improve the baseline detection rate by 20%, and the detection rate stays constant throughout the duration of the forecast. Assuming no additional interventions (ignoring Question 3), does that increase the amount of cumulative forecasted cases and deaths after six weeks? How does an increase in the detection rate affect the uncertainty in our estimates? Can you characterize the relationship between detection rate and our forecasts and their uncertainties, and comment on whether improving detection rates would provide decision-makers with better information (i.e., more accurate forecasts and/or narrower prediction intervals)?

```@example scenario3
# these new equations add I->D and H->R  to the model. 
# so :hosp is the detection rate, we assume all detected are hospitalized
sys_detect = add_accumulations(sys_detect, [I])
sys5 = add_accumulations(sys5, [I])
@unpack exp, conv, rec, hrec, hosp, death, ideath = sys5
ps = [
    exp => 0.2,
    conv => 0.2,
    rec => 0.6,
    hrec => 0.6,
    hosp => 0.2,
    death => 0.1,
    ideath => 0.1,
]
saveat = 0:tend
prob_baseline = ODEProblem(sys1_, u0init, (0.0, tend), ps)
probd = ODEProblem(sys5, u0init, (0.0, tend), ps)
sol_baseline = solve(prob_baseline; saveat)
sold = solve(probd; saveat)
plot(sol_baseline)
plot(sold)
```

```julia
# sweep over detection rates 
sols = []
u_detecs = 0:0.1:1
for x in u_detecs
    probd = remake(probd, p = [hosp => x])
    sold = solve(probd; saveat = sold.t)
    push!(sols, sold)
end

# demonstrate that the total infected count is strictly decreasing with increasing detection rate
is = map(x -> x[accumulation_I][end], sols)
plot(is)
@test issorted(is; rev = true)

# deaths decrease with increasing detection rate
ds = map(x -> x[D][end], sols)
plot(ds)
@test issorted(ds; rev = true)
```

```julia
# now show affect on uncertainty
get_uncertainty_forecast(probd, accumulation_I, 0:100,
                               [hosp => Uniform(0.0, 1.0), conv => Uniform(0.0, 1.0)],
                               6 * 7)

plot_uncertainty_forecast(probd, accumulation_I, 0:100,
                                   [
                                       hosp => Uniform(0.0, 1.0),
                                       conv => Uniform(0.0, 1.0),
                                   ],
                                   6 * 7)
plot_uncertainty_forecast(prob_baseline, accumulation_I, 0:100,
                                   [
                                       conv => Uniform(0.0, 1.0),
                                   ],
                                   6 * 7)

plot_uncertainty_forecast_quantiles(probd, D, 0:100,
                                   [
                                       hosp => Uniform(0.0, 1.0),
                                       conv => Uniform(0.0, 1.0),
                                   ],
                                   6 * 7)
plot_uncertainty_forecast_quantiles(prob_baseline, D, 0:100,
                                   [
                                       conv => Uniform(0.0, 1.0),
                                   ],
                                   6 * 7)
# this indicates that taking detection rate into account increases uncertainty in forecasts
```

> Compute the accuracy of the forecast assuming no mask mandate (ignoring Question 3) in the same way as you did in Question 1 and determine if improving the detection rate improves forecast accuracy.

### Question 5

> Convert the MechBayes SEIRHD model to an SIRHD model by removing the E compartment. Compute the same six-week forecast that you had done in Question 1a and compare the accuracy of the six-week forecasts with the forecasts done in Question 1a.

```julia
# no data to compare accuracy
sys_sirhd = add_accumulations(sys_sirhd, [I])

prob3 = ODEProblem(sys_sirhd, u0init, (0.0, 6 * 7.0), ps)
get_uncertainty_forecast(prob3, accumulation_I, 0:100, [exp => Uniform(0.0, 1.0)], 6 * 7)
```

```julia
plot_uncertainty_forecast(prob3, accumulation_I, 0:100, [exp => Uniform(0.0, 1.0)],
                          6 * 7)
```

```julia
get_uncertainty_forecast_quantiles(prob3, accumulation_I, 0:100,
                                   [exp => Uniform(0.0, 1.0)],
                                   6 * 7)
```

```julia
plot_uncertainty_forecast_quantiles(prob3, accumulation_I, 0:100,
                                    [exp => Uniform(0.0, 1.0)],
                                    6 * 7)
```

> Further modify the MechBayes SEIRHD model and do a model space exploration and model selection from the following models, based on comparing forecasts of cases and deaths to actual data: SEIRD, SEIRHD, and SIRHD models. Use data from April 1, 2020 – April 30, 2020 from the scenario location (Massachusetts) for fitting these models.  Then make out-of-sample forecasts from the same 6-week period from May 1 – June 15, 2020, and compare with actual data. Comment on the quality of the fit for each of these models.

```julia
sys_seirhd = add_accumulations(sys_seirhd, [I])
sys_seird = add_accumulations(sys_seird, [I])

prob_seirhd = ODEProblem(sys_seirhd, u0init, (0.0, 6 * 7.0), ps)
prob_seird = ODEProblem(sys_seird, u0init, (0.0, 6 * 7.0), ps)
prob_sirhd = ODEProblem(sys_sirhd, u0init, (0.0, 6 * 7.0), ps)
# no data to fit to
```

```julia
plot_uncertainty_forecast(prob_seirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)],
                          6 * 7)
plot_uncertainty_forecast(prob_seird, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)],
                          6 * 7)
plot_uncertainty_forecast(prob_sirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)],
                          6 * 7) #sus
```

```julia

get_uncertainty_forecast_quantiles(prob_seirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)],6 * 7)
get_uncertainty_forecast_quantiles(prob_seird, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)], 6 * 7)
get_uncertainty_forecast_quantiles(prob_sirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)], 6 * 7) #sus
```

```julia
plot_uncertainty_forecast_quantiles(prob_seirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)],6 * 7)
plot_uncertainty_forecast_quantiles(prob_seird, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)], 6 * 7)
plot_uncertainty_forecast_quantiles(prob_sirhd, accumulation_I, 0:100, [conv => Uniform(0.0, 1.0)], 6 * 7) #sus

```

> Do a 3-way structural model comparison between the SEIRD, SEIRHD, and SIRHD models.

```@example scenario3
AlgebraicPetri.ModelComparison.compare(seirhd, seirhd)
# todo
```

### https://github.com/SciML/EasyModelAnalysis.jl/issues/22

### Question 7

> What is the latest date we can impose a mandatory mask mandate over the next six weeks to ensure, with 90% probability, that cumulative deaths do not exceed 6000? Can you characterize the following relationship: for every day that we delay implementing a mask mandate, we expect cumulative deaths (over the six-week timeframe) to go up by X?
